#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include <boost/program_options.hpp>

#include "halco/common/iter_all.h"
#include "halco/hicann/v2/external.h"
#include "halco/hicann/v2/hicann.h"
#include "halco/hicann/v2/wafer.h"

#include "hal/HICANN/SynapseDriver.h"

#include "hal/Handle/FPGAHw.h"
#include "hal/backend/FPGABackend.h"
#include "hal/backend/HICANNBackend.h"

#include "redman/backend/Library.h"
#include "redman/resources/Wafer.h"
#include "redman/Policy.h"

#include "hwdb4cpp/hwdb4cpp.h"

#include "logger.h"


namespace C {
using namespace halco::hicann::v2;
using namespace halco::common;
}
namespace RR = redman::resources;
namespace Handle = HMF::Handle;
namespace Backend = HMF;
namespace po = boost::program_options;

// Define logger variable
log4cxx::LoggerPtr test_logger = log4cxx::Logger::getLogger("cake.digital_blacklisting");

// sets has_value = True if component not already touched
template <typename res>
void touch_component(boost::shared_ptr<res> component)
{
	if (!component->has_value())
		component->enable_all();
}

// checks if driver exists on HICANN v4
bool drv_exists(C::SynapseDriverOnHICANN syn_drv)
{
	// synapse rows of synase drivers 110 - 113 were removed on HICANN v4
	size_t drv_enum = syn_drv.toEnum();
	if (110 <= drv_enum && drv_enum < 114) {
		return false;
	}
	return true;
}

/**
 * @brief Write/read repeater_c with repeater settings.
 * @param hicann_handle Handle of tested HICANN
 * @param repeater_c Repeater coordinate
 * @param repeater_settings Repeater settings
 * @return false if mismatch or wrong configuration is read back, otherwise true.
 */
template <typename R, typename Co>
bool write_read_test_repeater(
    boost::shared_ptr<Handle::HICANNHw> const& hicann_handle,
    Co const& repeater_c,
    R const& repeater_settings)
{
	Backend::HICANN::set_repeater(*hicann_handle, repeater_c, repeater_settings);
	// catch special error (illegal repeater config is read back) and blacklist
	try {
		auto const read_repeater_settings =
		    Backend::HICANN::get_repeater(*hicann_handle, repeater_c);
		if (repeater_settings != read_repeater_settings) {
			return false;
		}
	} catch (const std::domain_error& e) {
		LOG4CXX_WARN(
		    test_logger, "Catched error during Repeater " << repeater_c << " test: " << e.what());
		return false;
	}
	return true;
}

/**
 * @brief Get associated repeater configuration container for repeater coordinate
 * @param repeater_c repeater coordinate
 * @return Associated halbe repeater
 */
template <typename Co>
auto get_repeater_container(Co const& repeater_c)
{
	if constexpr (repeater_c.size == C::VRepeaterOnHICANN::size) {
		return HMF::HICANN::VerticalRepeater();
	} else if constexpr (repeater_c.size == C::HRepeaterOnHICANN::size) {
		return HMF::HICANN::HorizontalRepeater();
	}
}

/**
 * @brief Get associated repeater directions for repeater coordinate
 * @param repeater_c repeater coordinate
 * @return Associated repeater directions
 */
template <typename Co>
auto get_direction(Co const& repeater_c)
{
	if constexpr (repeater_c.size == C::VRepeaterOnHICANN::size) {
		return std::array{C::top, C::bottom};
	} else if constexpr (repeater_c.size == C::HRepeaterOnHICANN::size) {
		return std::array{C::left, C::right};
	}
}

/**
 * @brief Write/read tests H/V repeater with all possible configurations and
 *  blacklist if a mismatch is found
 * @param hicann_handle Handle of tested HICANN
 * @param repeater_c Repeater coordinate
 * @param defects Redman hicann defects file where component gets blacklisted
 * @param rewrite_policy Redman policy to allow overwriting
 */
template <typename Co, typename D>
void test_repeater(
    boost::shared_ptr<Handle::HICANNHw> const& hicann_handle,
    Co const& repeater_c,
    D const& defects,
    auto const rewrite_policy)
{
	auto repeater_settings(get_repeater_container(repeater_c));
	auto direction(get_direction(repeater_c));
	for (auto ren = 0; ren < 4; ren++) {
		for (auto len = 0; len < 4; len++) {
			for (auto opt = 0; opt < 9; opt++) {
				repeater_settings.setRen(ren);
				repeater_settings.setLen(len);
				switch (opt) {
					case 0:
						repeater_settings.setIdle();
						break;
					case 1:
						repeater_settings.setForwarding(direction[0]);
						break;
					case 2:
						repeater_settings.setForwarding(direction[1]);
						break;
					case 3:
						repeater_settings.setInput(direction[0]);
						break;
					case 4:
						repeater_settings.setInput(direction[1]);
						break;
					// set output: not allowed that both directions are false
					case 5:
						repeater_settings.setOutput(direction[0], true);
						repeater_settings.setOutput(direction[1], false);
						break;
					case 6:
						repeater_settings.setOutput(direction[1], true);
						repeater_settings.setOutput(direction[0], false);
						break;
					case 7:
						repeater_settings.setOutput(direction[0], true);
						repeater_settings.setOutput(direction[1], true);
						break;
					case 8:
						repeater_settings.setLoopback();
						break;
				}
				// write and read values
				if (!write_read_test_repeater(hicann_handle, repeater_c, repeater_settings)) {
					// disable defect repeater
					defects->disable(repeater_c, rewrite_policy);
				}
			}
		}
	}
}

/**
 * @brief Repeatedly r/w test the synapse weights and decoders with the same values
 * A random number generator is used to generate different test values per component
 * @param array Synapse array coordinate which will be checked
 * @param generator Random number generator (seeded with stability_seed)
 * @param hicann_handle Handle of tested HICANN
 * @param number_of_repetitions How often the same value is tested per component
 * @param synapse_controller Synapse controller needed for dummy waits
 * @return false if unstable behaviour was found else true
 */
bool test_synapse_array_stability(
    C::SynapseArrayOnHICANN const& syn_array_c,
    std::mt19937& generator,
    boost::shared_ptr<Handle::HICANNHw> const& hicann_handle,
    size_t number_of_repetitions,
    HMF::HICANN::SynapseController const& synapse_controller)
{
	std::uniform_int_distribution<int> syn_weight_num(
	    HMF::HICANN::SynapseWeight::min, HMF::HICANN::SynapseWeight::max);
	// check weight stability
	for (auto const& syn_row_on_array_c : C::iter_all<C::SynapseRowOnArray>()) {
		C::SynapseRowOnHICANN const syn_row_c(syn_row_on_array_c, syn_array_c);
		// skip unavailable components
		if (!drv_exists(syn_row_c.toSynapseDriverOnHICANN())) {
			continue;
		}
		HMF::HICANN::WeightRow weight_row;
		// initialize each synapse with a random weight
		for (auto const& syn_column_c : C::iter_all<C::SynapseColumnOnHICANN>()) {
			// generate SynapseWeight objects to fill WeightRow with random weights
			HMF::HICANN::SynapseWeight syn_weight(syn_weight_num(generator));
			weight_row[syn_column_c] = syn_weight;
		}
		std::set<int> found_defect_numbers;
		for (size_t repetition = 0; repetition < number_of_repetitions; repetition++) {
			int defect_counter = 0;
			// use halbe backend functions to write and read
			Backend::HICANN::set_weights_row(
			    *hicann_handle, synapse_controller, syn_row_c, weight_row);
			HMF::HICANN::WeightRow const read_weight_row =
			    Backend::HICANN::get_weights_row(*hicann_handle, synapse_controller, syn_row_c);
			// compare to extract the number of defect synapses
			for (auto const& syn_column_c : C::iter_all<C::SynapseColumnOnHICANN>()) {
				if (weight_row[syn_column_c] != read_weight_row[syn_column_c]) {
					defect_counter++;
				}
			}
			found_defect_numbers.insert(defect_counter);
		}
		if (found_defect_numbers.size() != 1) {
			// unstable defect behaviour found
			std::stringstream out;
			out << "Unstable behaviour during weight stability test on " << syn_row_c << " on "
			    << syn_array_c << '\n';
			out << "Error counts: [";
			for (auto count : found_defect_numbers) {
				out << count << " ";
			}
			out << "]";
			LOG4CXX_INFO(test_logger, out.str());
			return false;
		}
	} // check weight stability
	// check decoder stability
	std::uniform_int_distribution<int> syn_dec_num(
	    HMF::HICANN::SynapseDecoder::min, HMF::HICANN::SynapseDecoder::max);
	for (auto const& syn_drv_c : C::iter_all<C::SynapseDriverOnHICANN>()) {
		// skip drivers on other synapse array
		if (syn_drv_c.toSynapseArrayOnHICANN() != syn_array_c) {
			continue;
		}
		// skip if driver is unavailable on HICANN v4
		if (!drv_exists(syn_drv_c)) {
			continue;
		}
		HMF::HICANN::DecoderDoubleRow decoder_double_row;
		// Initialize each decoder with random values
		for (auto const& row_number : C::iter_all<C::RowOnSynapseDriver>()) {
			HMF::HICANN::DecoderRow dec_row;
			for (auto const& syn_column_c : C::iter_all<C::SynapseColumnOnHICANN>()) {
				dec_row[syn_column_c] = HMF::HICANN::SynapseDecoder(syn_dec_num(generator));
			}
			decoder_double_row[row_number] = dec_row;
		}
		std::set<int> found_defect_numbers;
		for (size_t repetition = 0; repetition < number_of_repetitions; repetition++) {
			int defect_counter = 0;
			// write and read values
			Backend::HICANN::set_decoder_double_row(
			    *hicann_handle, synapse_controller, syn_drv_c, decoder_double_row);
			HMF::HICANN::DecoderDoubleRow const read_decoder_row =
			    Backend::HICANN::get_decoder_double_row(
			        *hicann_handle, synapse_controller, syn_drv_c);
			// compare values
			for (auto const& row_number : C::iter_all<C::RowOnSynapseDriver>()) {
				for (auto const& syn_column_c : C::iter_all<C::SynapseColumnOnHICANN>()) {
					if (decoder_double_row[row_number][syn_column_c] !=
					    read_decoder_row[row_number][syn_column_c]) {
						defect_counter++;
					}
				}
			}
			found_defect_numbers.insert(defect_counter);
		}
		if (found_defect_numbers.size() != 1) {
			// unstable defect behaviour found
			std::stringstream out;
			out << "Unstable behaviour during decoder stability test on decoder double row of "
			    << syn_drv_c << " on " << syn_array_c << '\n';
			out << "Error counts: [";
			for (auto count : found_defect_numbers) {
				out << count << " ";
			}
			out << "]";
			LOG4CXX_INFO(test_logger, out.str());
			return false;
		}
	} // check decoder stability
	return true;
}

int main(int argc, char* argv[])
{
	size_t wafer;
	size_t hicann;
	std::string hwdb_path;
	size_t pll_freq;
	size_t jtag_frequency;
	bool allow_rewrite;
	std::string input_backend_path;
	std::string output_backend_path;
	bool highspeed;
	std::vector<size_t> seeds;
	size_t stability_seed;
	size_t stability_repetitions;

	// parse arguments from command line
	po::options_description options("Allowed options");
	options.add_options()("help", "produce help message")(
	    "wafer,w", po::value<size_t>(&wafer)->required(), "set wafer")(
	    "hicann,h", po::value<size_t>(&hicann)->required(), "set hicann")(
	    "path,p", po::value<std::string>(&hwdb_path)->default_value("/wang/data/bss-hwdb/db.yaml"),
	    "set path to hwdb")(
	    "pll", po::value<size_t>(&pll_freq)->default_value(125), "set PLL frequency")(
	    "jtag_frequency,j", po::value<size_t>(&jtag_frequency)->default_value(10e6), "set JTAG frequency in Hz")(
	    "allow_rewrite", po::value<bool>(&allow_rewrite)->default_value(true),
	    "allow to rewrite already disabled values")(
	    "input_backend_path", po::value<std::string>(&input_backend_path)->default_value("./"),
	    "path to communication test results")(
	    "output_backend_path", po::value<std::string>(&output_backend_path)->default_value("./"),
	    "path where blacklisting results of this test are stored. If blacklisting information is already present,"
	    " it gets adapted and is used to control the test")(
	    "highspeed", po::value<bool>(&highspeed)->default_value(1), "use highspeed otherwise JTAG")(
	    "seeds", po::value<std::vector<size_t> >(&seeds)->multitoken()->required(), "used seeds")(
	    "stability_seed", po::value<size_t>(&stability_seed)->default_value(10),
	    "used seed for synapse stability test")(
	    "stability_repetitions", po::value<size_t>(&stability_repetitions)->default_value(20),
	    "number of repetitions in the synapse stability test");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, options), vm);
	if (vm.count("help")) {
		std::cout << options << "\n";
		return EXIT_SUCCESS;
	}
	po::notify(vm);

	// Initialize logger and set log level
	logger_default_config(log4cxx::Level::getInfo());

	redman::switch_mode::type const rewrite_policy = allow_rewrite ? redman::switch_mode::NONTHROW : redman::switch_mode::THROW;

	// parse HICANN to be tested, ignore all others
	C::Wafer const wafer_c(wafer);
	C::HICANNOnWafer const hicann_c{C::Enum(hicann)};
	C::FPGAGlobal const fpga_c(hicann_c.toFPGAOnWafer(), wafer_c);

	LOG4CXX_INFO(test_logger, "Test HICANN " << hicann << " on Wafer " << wafer);

	// instantiate redman
	// input_backend contains the blacklisting results of the commtest
	// output_backend contains the results of this test
	// also used during iterative tests -> backend is loaded and used
	// to ignore already blacklisted components
	auto const lib = redman::backend::loadLibrary("libredman_xml.so");
	auto const input_backend = redman::backend::loadBackend(lib);
	auto const output_backend = redman::backend::loadBackend(lib);
	if (!input_backend || !output_backend) {
		throw std::runtime_error("unable to load xml backend");
	}
	// input_backend contains the communication test results, which are used to establish the
	// connection to the reticle (redman_wafer, redman_fpga)
	input_backend->config("path", input_backend_path);
	// output_backend is used to store the results. If blacklisting information is already present
	// it gets adapted (redman_hicann). Additionally it is used to control the test (redman_hicann_previous_test)
	output_backend->config("path", output_backend_path);
	input_backend->init();
	output_backend->init();
	RR::HicannWithBackend redman_hicann(output_backend, C::HICANNGlobal(hicann_c, wafer_c));
	// redman_hicann_previous_test is used to check if component was already blacklisted in previous tests
	// -> tests are not skipped if higher order component is blacklisted in same run
	RR::HicannWithBackend const redman_hicann_previous_test(output_backend, C::HICANNGlobal(hicann_c, wafer_c));
	RR::WaferWithBackend const redman_wafer(input_backend, wafer_c);
	RR::FpgaWithBackend const redman_fpga(input_backend, fpga_c);

	std::set<C::HICANNOnDNC> available_hicanns; // physically present (wafer vs. cube)
	std::set<C::HICANNOnDNC>
	    highspeed_hicanns; // to be accessed via highspeed connection (jtag otherwise)
	std::set<C::HICANNOnDNC> usable_hicanns; // if not in this set, HICANN is not talked to at all

	// load hwdb and set handle parameters
	hwdb4cpp::database hwdb;
	hwdb.load(hwdb_path);
	hwdb4cpp::WaferEntry const hwdb_wafer = hwdb.get_wafer_entry(wafer_c);

	auto it = hwdb_wafer.fpgas.find(fpga_c);
	if (it == hwdb_wafer.fpgas.end()) {
		throw std::runtime_error("Couldn't find FPGA in database; key =" + std::to_string(fpga_c));
	}
	auto const fpga_ip = it->second.ip;

	for (auto h : C::iter_all<C::HICANNOnDNC>()) {
		auto const hicann_global = C::HICANNGlobal(h.toHICANNOnWafer(fpga_c), wafer_c);
		if (hwdb.has_hicann_entry(hicann_global)) {
			available_hicanns.insert(h);
		}
	}

	// check redman data for all HICANNs on reticle
	// if HICANN is jtag-only it is only used for routing
	//  -> some tests and associated resets can be skipped
	bool test_additional_components = false;
	bool reset_weights_and_decoders = false;
	for (auto h : C::iter_all<C::HICANNOnDNC>()) {
		C::HICANNOnWafer h_on_wafer = h.toHICANNOnWafer(fpga_c);
		if (redman_wafer.has(h_on_wafer)) {
			// add hicann to usable hicanns
			usable_hicanns.insert(h);
			LOG4CXX_INFO(
			    test_logger, "HICANN " << h_on_wafer.toEnum().value() << " on Wafer " << wafer
			                           << " added to usable HICANNs.");
		} else if (h_on_wafer == hicann_c) {
			// currently used HICANN is blacklisted
			LOG4CXX_INFO(
			    test_logger,
			    "Tested HICANN " << hicann << " on Wafer " << wafer << " is already blacklisted.");
			return EXIT_SUCCESS;
		}
		// disable highspeed if blacklisted or user input
		if (redman_fpga.hslinks()->has(h.toHighspeedLinkOnDNC()) && highspeed) {
			highspeed_hicanns.insert(h);
			if (hicann_c == h_on_wafer) {
				test_additional_components = true;
				reset_weights_and_decoders = true;
			}
			LOG4CXX_INFO(
			    test_logger, "HICANN " << h_on_wafer.toEnum().value() << " on Wafer " << wafer
			                           << " added to highspeed HICANNs.");
		}
	}

	auto const dnc = C::DNCGlobal(fpga_c.toDNCOnWafer(), wafer_c).toDNCOnFPGA();

	// instantiate handle and reset
	Handle::FPGAHw::HandleParameter handle_param{fpga_c,
	                                             fpga_ip,
	                                             dnc,
	                                             available_hicanns,
	                                             highspeed_hicanns,
	                                             usable_hicanns,
	                                             hwdb_wafer.setup_type,
	                                             hwdb_wafer.macu,
	                                             C::JTAGFrequency(jtag_frequency)};

	HMF::Handle::FPGAHw fpga_handle(handle_param);
	auto hicann_handle = fpga_handle.get(C::HICANNGlobal(hicann_c, wafer_c));
	HMF::FPGA::Reset r;
	r.PLL_frequency = pll_freq;
	HMF::FPGA::reset(fpga_handle, r);

	// reprogram all usable HICANNs of FPGA and optionally set weight and decoder values to 0
	HMF::FPGA::init(fpga_handle, reset_weights_and_decoders);

	// generate integer distributions to define the range of the random values
	std::uniform_int_distribution<int> true_false(0, 1);
	std::uniform_int_distribution<int> drv_dec_num(
	    HMF::HICANN::DriverDecoder::min, HMF::HICANN::DriverDecoder::max);
	std::uniform_int_distribution<int> syn_weight_num(
	    HMF::HICANN::SynapseWeight::min, HMF::HICANN::SynapseWeight::max);
	std::uniform_int_distribution<int> syn_dec_num(
	    HMF::HICANN::SynapseDecoder::min, HMF::HICANN::SynapseDecoder::max);
	std::uniform_int_distribution<int> l1_address(
	    HMF::HICANN::L1Address::min, HMF::HICANN::L1Address::max);
	std::uniform_int_distribution<int> number2(0, 2);
	std::uniform_int_distribution<int> number3(0, 3);
	std::uniform_int_distribution<int> number4(0, 4);
	std::uniform_int_distribution<int> number5(0, 5);
	std::uniform_int_distribution<int> number7(0, 7);
	std::uniform_int_distribution<int> number8(0, 8);
	std::uniform_int_distribution<int> number9(0, 9);
	std::uniform_int_distribution<int> number15(0, 15);
	std::uniform_int_distribution<int> number223(0, 223);
	std::uniform_int_distribution<int> number4bit(0, 15);
	std::uniform_int_distribution<int> number6bit(0, 63);
	std::uniform_int_distribution<int> number8bit(0, 255);
	std::uniform_int_distribution<int> number10bit(0, 1023);
	std::uniform_int_distribution<int> number16bit(0, 65535);
	std::uniform_int_distribution<unsigned long int> number32bit(0, 4294967295);
	std::uniform_int_distribution<int> numberRepBlock(
	    0, (1 << halco::hicann::v2::TestPortOnRepeaterBlock::end) - 1);
	std::uniform_int_distribution<int> numberstpcap(
	    0, (1 << HMF::HICANN::SynapseDriver::num_cap) - 1);

	// Synapse controller needed for dummy waits. Writing default constructed
	// values to hardware is valid, since configuration register is not altered
	// in this test.
	HMF::HICANN::SynapseController const synapse_controller;

	// Once touch all tested components to set has_value = True
	// Used to distinguish between tested and not tested components
	// If error occurs during test xml file is not saved anyway
	if (test_additional_components) {
		touch_component<RR::components::SynapseDrivers>(redman_hicann.drivers());
		touch_component<RR::components::SynapseRows>(redman_hicann.synapserows());
		touch_component<RR::components::Synapses>(redman_hicann.synapses());
		touch_component<RR::components::Analogs>(redman_hicann.analogs());
		touch_component<RR::components::BackgroundGenerators>(redman_hicann.backgroundgenerators());
		touch_component<RR::components::DNCMergers>(redman_hicann.dncmergers());
		touch_component<RR::components::Mergers0>(redman_hicann.mergers0());
		touch_component<RR::components::Mergers1>(redman_hicann.mergers1());
		touch_component<RR::components::Mergers2>(redman_hicann.mergers2());
		touch_component<RR::components::Mergers3>(redman_hicann.mergers3());
		touch_component<RR::components::SynapseArrays>(redman_hicann.synapsearrays());
	}
	touch_component<RR::components::FGBlocks>(redman_hicann.fgblocks());
	touch_component<RR::components::SynapseSwitches>(redman_hicann.synapseswitches());
	touch_component<RR::components::CrossbarSwitches>(redman_hicann.crossbarswitches());
	touch_component<RR::components::VRepeaters>(redman_hicann.vrepeaters());
	touch_component<RR::components::HRepeaters>(redman_hicann.hrepeaters());
	for (auto seed : seeds) {
		LOG4CXX_INFO(test_logger, "Using seed " << seed);
		// set seed for random values
		std::mt19937 generator(seed);
		// handle errors in single components -> init reticle and return EXIT_FAILURE
		// HICANN with defect component has to be checked manually
		try {
			// skip some tests for HICANNs without Highspeed
			if (test_additional_components) {
				// check set_Synapse_Driver
				// iterate Hicann Synapse Driver
				for (auto syn_drv_c : C::iter_all<C::SynapseDriverOnHICANN>()) {
					if (!drv_exists(syn_drv_c)) {
						continue;
					}
					// if component already blacklisted continue
					if (!redman_hicann_previous_test.drivers()->has(syn_drv_c))
						continue;
					LOG4CXX_INFO(
					    test_logger, "Test component: " << syn_drv_c << " on HICANN " << hicann
									    << " with seed " << seed);
					// configure syn_drv with random values
					HMF::HICANN::SynapseDriver syn_drv;
					switch (number5(generator)) {
						case 0:
							syn_drv.disable();
							break;
						case 1:
							syn_drv.set_l1();
							break;
						case 2:
							syn_drv.set_l1_mirror();
							break;
						case 3:
							syn_drv.set_mirror();
							break;
						case 4:
							syn_drv.set_mirror_only();
							break;
						case 5:
							syn_drv.set_listen();
							break;
					}
					switch (number2(generator)) {
						case 0:
							syn_drv.disable_stp();
							break;
						case 1:
							syn_drv.set_stf();
							break;
						case 2:
							syn_drv.set_std();
							break;
					}
					syn_drv.stp_cap = numberstpcap(generator);
					// configure syn:drv::mRowConfig with random values
					// mRowConfig is element of array<RowConfig, num_rows>
					for (auto row_on_drv : C::iter_all<C::RowOnSynapseDriver>()) {
						syn_drv[row_on_drv].set_gmax(number3(generator));
						syn_drv[row_on_drv].set_gmax_div(C::left, number15(generator));
						syn_drv[row_on_drv].set_gmax_div(C::right, number15(generator));
						syn_drv[row_on_drv].set_syn_in(C::left, true_false(generator));
						syn_drv[row_on_drv].set_syn_in(C::right, true_false(generator));
						syn_drv[row_on_drv].set_decoder(
						    C::top, HMF::HICANN::DriverDecoder(drv_dec_num(generator)));
						syn_drv[row_on_drv].set_decoder(
						    C::bottom, HMF::HICANN::DriverDecoder(drv_dec_num(generator)));
					}
					// use halbe backend functions to write and read
					Backend::HICANN::set_synapse_driver(*hicann_handle, synapse_controller, syn_drv_c, syn_drv);
					HMF::HICANN::SynapseDriver const read_synapse_driver =
					    Backend::HICANN::get_synapse_driver(*hicann_handle, synapse_controller, syn_drv_c);
					// compare and blacklist the associated coordinate
					// SynapseDriver::operator== compares all driver variables
					// not used here to distinguish error in whole driver or only in row on driver
					if ((syn_drv.is_l1() != read_synapse_driver.is_l1()) ||
					    (syn_drv.is_l1_mirror() != read_synapse_driver.is_l1_mirror()) ||
					    (syn_drv.is_mirror() != read_synapse_driver.is_mirror()) ||
					    (syn_drv.is_mirror_only() != read_synapse_driver.is_mirror_only()) ||
					    (syn_drv.is_listen() != read_synapse_driver.is_listen()) ||
					    (syn_drv.is_stf() != read_synapse_driver.is_stf()) ||
					    (syn_drv.stp_cap != read_synapse_driver.stp_cap) ||
					    (syn_drv.is_std() != read_synapse_driver.is_std())) {
						// error in driver => disable driver
						redman_hicann.drivers()->disable(syn_drv_c, rewrite_policy);
					}
					// check row on driver
					for (auto row_on_drv : C::iter_all<C::RowOnSynapseDriver>()) {
						if (syn_drv[row_on_drv] != read_synapse_driver[row_on_drv]) {
							// disable defect SynapseRowOnHICANN
							redman_hicann.synapserows()->disable(
							    C::SynapseRowOnHICANN(syn_drv_c, row_on_drv), rewrite_policy);
						}
					}
				} // check set_Synapse_Driver

				// check set_weights_row
				for (auto const& syn_row_c : C::iter_all<C::SynapseRowOnHICANN>()) {
					if (!drv_exists(syn_row_c.toSynapseDriverOnHICANN())) {
						continue;
					}
					// if component already blacklisted continue
					if (!redman_hicann_previous_test.synapserows()->has(syn_row_c))
						continue;
					LOG4CXX_INFO(
					    test_logger, "Test component: " << syn_row_c << " on HICANN " << hicann
									    << " with seed " << seed);
					HMF::HICANN::WeightRow weight_row;
					// initialize weight which is array of SynapseWeight with size of Synapse Column
					// (256)
					for (auto const& syn_column_c : C::iter_all<C::SynapseColumnOnHICANN>()) {
						// Generate object SynapseWeight to fill array WeightRow
						HMF::HICANN::SynapseWeight syn_weight(syn_weight_num(generator));
						weight_row[syn_column_c] = syn_weight;
					}
					// use halbe backend functions to write and read
					Backend::HICANN::set_weights_row(*hicann_handle, synapse_controller, syn_row_c, weight_row);
					HMF::HICANN::WeightRow const read_weight_row =
					    Backend::HICANN::get_weights_row(*hicann_handle, synapse_controller, syn_row_c);

					// compare and blacklist the associated coordinate of defect synapses
					int error_counter = 0;
					for (auto const& syn_column_c : C::iter_all<C::SynapseColumnOnHICANN>()) {
						if (weight_row[syn_column_c] != read_weight_row[syn_column_c]) {
							// disable defect SynapseOnHICANN
							redman_hicann.synapses()->disable(
							    C::SynapseOnHICANN(syn_column_c, syn_row_c), rewrite_policy);
							error_counter++;
						}
					}
					// if complete row is damaged disable SynapseRowOnHICANN
					if (error_counter == C::SynapseColumnOnHICANN::max) {
						redman_hicann.synapserows()->disable(syn_row_c, rewrite_policy);
					}
				} // check set_weights_row

				// check set decoder_double_row
				for (auto const& syn_drv_c : C::iter_all<C::SynapseDriverOnHICANN>()) {
					if (!drv_exists(syn_drv_c)) {
						continue;
					}
					// if component already blacklisted continue
					if (!redman_hicann_previous_test.drivers()->has(syn_drv_c))
						continue;
					LOG4CXX_INFO(
					    test_logger, "Test component: " << syn_drv_c << " on HICANN " << hicann
									    << " with seed " << seed);
					HMF::HICANN::DecoderDoubleRow decoder_double_row;
					// Fill with random values
					for (auto const& row_number : C::iter_all<C::RowOnSynapseDriver>()) {
						HMF::HICANN::DecoderRow dec_row;
						for (auto const& syn_column_c : C::iter_all<C::SynapseColumnOnHICANN>()) {
							dec_row[syn_column_c] = HMF::HICANN::SynapseDecoder(syn_dec_num(generator));
						}
						decoder_double_row[row_number] = dec_row;
					}
					// write and read values
					Backend::HICANN::set_decoder_double_row(
					    *hicann_handle, synapse_controller, syn_drv_c, decoder_double_row);
					HMF::HICANN::DecoderDoubleRow const read_decoder_row =
					    Backend::HICANN::get_decoder_double_row(*hicann_handle, synapse_controller, syn_drv_c);
					// compare values
					for (auto const& row_number : C::iter_all<C::RowOnSynapseDriver>()) {
						for (auto const& syn_column_c : C::iter_all<C::SynapseColumnOnHICANN>()) {
							if (decoder_double_row[row_number][syn_column_c] !=
							    read_decoder_row[row_number][syn_column_c]) {
								// disable defect Synapse
								redman_hicann.synapses()->disable(
								    C::SynapseOnHICANN(
									C::SynapseRowOnHICANN(syn_drv_c, row_number), syn_column_c),
								    rewrite_policy);
							}
						}
					}
				} // check set decoder_double_row

				// check set_analog
				HMF::HICANN::Analog analog;
				// Fill with random values
				for (auto const& side : C::iter_all<C::AnalogOnHICANN>()) {
					// if component already blacklisted continue
					if (!redman_hicann_previous_test.analogs()->has(side))
						continue;
					LOG4CXX_INFO(
					    test_logger,
					    "Test component: " << side << " on HICANN " << hicann << " with seed " << seed);
					// set functions disable all other bits and enable their corresponding bit and call
					// the enable function for current side
					// =>call random set function
					switch (number9(generator)) {
						case 0:
							analog.set_dll_voltage(side);
							break;
						case 1:
							analog.set_fg_left(side);
							break;
						case 2:
							analog.set_fg_right(side);
							break;
						case 3: // not available on bottom analog readout
							analog.set_fireline_neuron0(C::AnalogOnHICANN(0));
							break;
						case 4:
							analog.set_membrane_bot_even(side);
							break;
						case 5:
							analog.set_membrane_top_even(side);
							break;
						case 6:
							analog.set_membrane_bot_odd(side);
							break;
						case 7:
							analog.set_membrane_top_odd(side);
							break;
						case 8:
							analog.set_none(side);
							break;
						case 9:
							analog.set_preout(side);
							break;
					}
					// also check to disable after set
					if (true_false(generator)) {
						analog.disable(side);
					}
				}
				// write and read values
				Backend::HICANN::set_analog(*hicann_handle, analog);
				HMF::HICANN::Analog const read_analog = Backend::HICANN::get_analog(*hicann_handle);
				// compare values
				for (auto const& side : C::iter_all<C::AnalogOnHICANN>()) {
					// catch possible error during analog test and blacklist if necessary
					try {
						if (analog.get_fg_left(side) != read_analog.get_fg_left(side) ||
						    analog.get_fg_right(side) != read_analog.get_fg_right(side) ||
						    analog.get_fireline_neuron0(side) !=
							read_analog.get_fireline_neuron0(side) ||
						    analog.get_membrane_bot_even(side) !=
							read_analog.get_membrane_bot_even(side) ||
						    analog.get_membrane_top_even(side) !=
							read_analog.get_membrane_top_even(side) ||
						    analog.get_membrane_bot_odd(side) !=
							read_analog.get_membrane_bot_odd(side) ||
						    analog.get_membrane_top_odd(side) !=
							read_analog.get_membrane_top_odd(side) ||
						    analog.get_none(side) != read_analog.get_none(side) ||
						    analog.get_preout(side) != read_analog.get_preout(side) ||
						    analog.enabled(side) != read_analog.enabled(side)) {
							// disable  Analog side
							redman_hicann.analogs()->disable(side, rewrite_policy);
						}
					} catch (const std::logic_error& e) {
						LOG4CXX_ERROR(
						    test_logger, "Error during Analog " << side << " test on HICANN " << hicann
											<< ": " << e.what());
						LOG4CXX_INFO(
						    test_logger, "Blacklist Analog:" << side << " on HICANN " << hicann);
						// blacklist component
						redman_hicann.analogs()->disable(side, rewrite_policy);
					}
				} // check set_analog

				// check set_background_generator
				// skip if all background generators are already disabled
				if (redman_hicann_previous_test.backgroundgenerators()->available()) {
					LOG4CXX_INFO(
					    test_logger,
					    "Test component: BackgroundGeneratorArray on HICANN " << hicann << " with seed " << seed);
					HMF::HICANN::BackgroundGeneratorArray backgroundgeneratorarray;
					// Fill BackgroundGeneratorArray with random values
					// use one seed since seed not gettable from hardware but check if it was delivered to
					// the hardware
					int seed_value = number16bit(generator);
					for (auto const& background_generator_c :
					     C::iter_all<C::BackgroundGeneratorOnHICANN>()) {
						HMF::HICANN::BackgroundGenerator backgroundgenerator;
						backgroundgenerator.enable(true_false(generator));
						backgroundgenerator.random(true_false(generator));
						backgroundgenerator.seed(seed_value);
						backgroundgenerator.period(number16bit(generator));
						backgroundgenerator.address(HMF::HICANN::L1Address(l1_address(generator)));
						// fill array with generated values
						backgroundgeneratorarray[background_generator_c] = backgroundgenerator;
					}
					// write and read values
					Backend::HICANN::set_background_generator(*hicann_handle, backgroundgeneratorarray);
					HMF::HICANN::BackgroundGeneratorArray const read_backgroundgeneratorarray =
					    Backend::HICANN::get_background_generator(*hicann_handle);
					// disable defect BackgroundGenerators
					for (auto const& background_generator_c :
					     C::iter_all<C::BackgroundGeneratorOnHICANN>()) {
						// seed neglected in == operator
						if (!(backgroundgeneratorarray[background_generator_c] ==
						      read_backgroundgeneratorarray[background_generator_c]) ||
						    !(backgroundgeneratorarray[background_generator_c].seed() ==
						      read_backgroundgeneratorarray[background_generator_c].seed())) {
							redman_hicann.backgroundgenerators()->disable(background_generator_c, rewrite_policy);
						}
					}
				}

				// check set_dnc_merger
				// skip if all DNCMerger are already disabled
				if (redman_hicann_previous_test.dncmergers()->available()) {
					LOG4CXX_INFO(
					    test_logger,
					    "Test component: DNCMergerLine on HICANN " << hicann << " with seed " << seed);
					HMF::HICANN::DNCMergerLine merger_line;
					for (auto const& merger_c : C::iter_all<C::DNCMergerOnHICANN>()) {
						// possible values of Merger: RIGHT_ONLY=0, LEFT_ONLY=1, MERGE=2 or use
						// HMF::Merge::MERGE
						HMF::HICANN::DNCMerger merger(number2(generator), true_false(generator));
						merger_line[merger_c] = merger;
					}
					// Loopback option (only possible in one direction)
					for (size_t merger_group = 0; merger_group < C::DNCMergerOnHICANN::size / 2;
					     merger_group++) {
						C::DNCMergerOnHICANN left_merger(merger_group * 2);
						C::DNCMergerOnHICANN right_merger(merger_group * 2 + 1);
						switch (number2(generator)) {
							case 0:
								// left merger loopback
								merger_line[left_merger].loopback = true;
								merger_line[right_merger].loopback = false;
								break;
							case 1:
								// right merger loopback
								merger_line[left_merger].loopback = false;
								merger_line[right_merger].loopback = true;
								break;
							case 2:
								// no loopback
								merger_line[left_merger].loopback = false;
								merger_line[right_merger].loopback = false;
								break;
						}
					}

					// write and read values
					Backend::HICANN::set_dnc_merger(*hicann_handle, merger_line);
					HMF::HICANN::DNCMergerLine const read_merger_line =
					    Backend::HICANN::get_dnc_merger(*hicann_handle);
					// disable defect dnc_merger
					for (auto const& merger_c : C::iter_all<C::DNCMergerOnHICANN>()) {
						if (merger_line[merger_c] != read_merger_line[merger_c]) {
							redman_hicann.dncmergers()->disable(merger_c, rewrite_policy);
						}
					}
				}

				// check set_merger_tree
				LOG4CXX_INFO(
				    test_logger,
				    "Test component: MergerTree on HICANN " << hicann << " with seed " << seed);
				HMF::HICANN::MergerTree merger_tree;
				for (auto const& merger0 : C::iter_all<C::Merger0OnHICANN>()) {
					HMF::HICANN::Merger merger(number2(generator), true_false(generator));
					merger_tree[merger0] = merger;
				}
				for (auto const& merger1 : C::iter_all<C::Merger1OnHICANN>()) {
					HMF::HICANN::Merger merger(number2(generator), true_false(generator));
					merger_tree[merger1] = merger;
				}
				for (auto const& merger2 : C::iter_all<C::Merger2OnHICANN>()) {
					HMF::HICANN::Merger merger(number2(generator), true_false(generator));
					merger_tree[merger2] = merger;
				}
				for (auto const& merger3 : C::iter_all<C::Merger3OnHICANN>()) {
					HMF::HICANN::Merger merger(number2(generator), true_false(generator));
					merger_tree[merger3] = merger;
				}
				// write and read values
				Backend::HICANN::set_merger_tree(*hicann_handle, merger_tree);
				HMF::HICANN::MergerTree const read_merger_tree =
				    Backend::HICANN::get_merger_tree(*hicann_handle);
				// disable defect merger
				for (auto const& merger0 : C::iter_all<C::Merger0OnHICANN>()) {
					if (merger_tree[merger0] != read_merger_tree[merger0]) {
						redman_hicann.mergers0()->disable(merger0, rewrite_policy);
					}
				}
				for (auto const& merger1 : C::iter_all<C::Merger1OnHICANN>()) {
					if (merger_tree[merger1] != read_merger_tree[merger1]) {
						redman_hicann.mergers1()->disable(merger1, rewrite_policy);
					}
				}
				for (auto const& merger2 : C::iter_all<C::Merger2OnHICANN>()) {
					if (merger_tree[merger2] != read_merger_tree[merger2]) {
						redman_hicann.mergers2()->disable(merger2, rewrite_policy);
					}
				}
				// loop not necessary 1 value
				for (auto const& merger3 : C::iter_all<C::Merger3OnHICANN>()) {
					if (merger_tree[merger3] != read_merger_tree[merger3]) {
						redman_hicann.mergers3()->disable(merger3, rewrite_policy);
					}
				}

				// check synapse control register
				for (auto const& synarray : C::iter_all<C::SynapseArrayOnHICANN>()) {
					if (!redman_hicann_previous_test.synapsearrays()->has(synarray))
						continue;
					LOG4CXX_INFO(
					    test_logger, "Test synapse control register on " << synarray
									    << " on HICANN " << hicann
									    << " with seed " << seed);

					HMF::HICANN::SynapseControlRegister ctrl_reg;

					ctrl_reg.sca = true_false(generator);
					ctrl_reg.scc = true_false(generator);
					ctrl_reg.without_reset = true_false(generator);
					ctrl_reg.sel = HMF::HICANN::SynapseSel(number7(generator));
					ctrl_reg.row = C::SynapseRowOnArray(number223(generator));
					ctrl_reg.last_row = C::SynapseRowOnArray(number223(generator));
					// set newcmd to false such that the hardware stays idle and does not
					// perform any commands
					ctrl_reg.newcmd = false;
					ctrl_reg.continuous = true_false(generator);
					ctrl_reg.encr = true_false(generator);
					switch (number8(generator)) {
						case 0:
							ctrl_reg.cmd = HMF::HICANN::SynapseControllerCmd::IDLE;
							break;
						case 1:
							ctrl_reg.cmd = HMF::HICANN::SynapseControllerCmd::READ;
							break;
						case 2:
							ctrl_reg.cmd = HMF::HICANN::SynapseControllerCmd::START_RDEC;
							break;
						case 3:
							ctrl_reg.cmd = HMF::HICANN::SynapseControllerCmd::WRITE;
							break;
						case 4:
							ctrl_reg.cmd = HMF::HICANN::SynapseControllerCmd::WDEC;
							break;
						case 5:
							ctrl_reg.cmd = HMF::HICANN::SynapseControllerCmd::RDEC;
							break;
						case 6:
							ctrl_reg.cmd = HMF::HICANN::SynapseControllerCmd::START_READ;
							break;
						case 7:
							ctrl_reg.cmd = HMF::HICANN::SynapseControllerCmd::CLOSE_ROW;
							break;
						case 8:
							ctrl_reg.cmd = HMF::HICANN::SynapseControllerCmd::RST_CORR;
							break;
					}

					// write and read values
					Backend::HICANN::set_syn_ctrl(*hicann_handle, synarray, ctrl_reg);
					try {
						HMF::HICANN::SynapseControlRegister read_ctrl_reg =
						    Backend::HICANN::get_syn_ctrl(*hicann_handle, synarray);

						if (ctrl_reg != read_ctrl_reg) {
							// disable whole synapsearray
							redman_hicann.synapsearrays()->disable(synarray, rewrite_policy);
						}
					} catch (const std::overflow_error& e) {
						LOG4CXX_WARN(
						    test_logger, "Catched error during synapse control register test on "
						                     << synarray << ": " << e.what());
						// disable whole synapsearray
						redman_hicann.synapsearrays()->disable(synarray, rewrite_policy);
					}
				}

				// check synapse configuration register
				for (auto const& synarray : C::iter_all<C::SynapseArrayOnHICANN>()) {
					if (!redman_hicann_previous_test.synapsearrays()->has(synarray))
						continue;
					LOG4CXX_INFO(
					    test_logger, "Test synapse configuration register on " << synarray
									    << " on HICANN " << hicann
									    << " with seed " << seed);

					HMF::HICANN::SynapseConfigurationRegister cnfg_reg;

					cnfg_reg.synarray_timings.write_delay =
						HMF::HICANN::SynapseWriteDelay(number3(generator));
					cnfg_reg.synarray_timings.output_delay =
						HMF::HICANN::SynapseOutputDelay(number4bit(generator));
					cnfg_reg.synarray_timings.setup_precharge =
						HMF::HICANN::SynapseSetupPrecharge(number4bit(generator));
					cnfg_reg.synarray_timings.enable_delay =
						HMF::HICANN::SynapseEnableDelay(number4bit(generator));
					cnfg_reg.dllresetb =
						HMF::HICANN::SynapseDllresetb(number3(generator));
					cnfg_reg.gen =
						HMF::HICANN::SynapseGen(number4bit(generator));
					cnfg_reg.pattern0.aa = true_false(generator);
					cnfg_reg.pattern0.ac = true_false(generator);
					cnfg_reg.pattern0.ca = true_false(generator);
					cnfg_reg.pattern0.cc = true_false(generator);
					cnfg_reg.pattern1.aa = true_false(generator);
					cnfg_reg.pattern1.ac = true_false(generator);
					cnfg_reg.pattern1.ca = true_false(generator);
					cnfg_reg.pattern1.cc = true_false(generator);

					// write and read values
					Backend::HICANN::set_syn_cnfg(*hicann_handle, synarray, cnfg_reg);
					HMF::HICANN::SynapseConfigurationRegister read_cnfg_reg =
						Backend::HICANN::get_syn_cnfg(*hicann_handle, synarray);


					if (cnfg_reg != read_cnfg_reg) {
						// disable whole synapsearray
						redman_hicann.synapsearrays()->disable(synarray, rewrite_policy);
					}
				}

				// check STDP LUT register
				for (auto const& synarray : C::iter_all<C::SynapseArrayOnHICANN>()) {
					if (!redman_hicann_previous_test.synapsearrays()->has(synarray))
						continue;
					LOG4CXX_INFO(
					    test_logger, "Test STDP LUT register on " << synarray
									    << " on HICANN " << hicann
									    << " with seed " << seed);

					HMF::HICANN::STDPLUT stdp_lut;

					for (uint8_t ii = HMF::HICANN::SynapseWeight::min;
							ii < HMF::HICANN::SynapseWeight::end; ++ii)
					{
						typedef HMF::HICANN::SynapseWeight weight_t;
						stdp_lut.causal[weight_t(ii)] = weight_t(number4bit(generator));
						stdp_lut.acausal[weight_t(ii)] = weight_t(number4bit(generator));
						stdp_lut.combined[weight_t(ii)] = weight_t(number4bit(generator));
					}

					// write and read values
					Backend::HICANN::set_stdp_lut(*hicann_handle, synarray, stdp_lut);
					HMF::HICANN::STDPLUT read_stdp_lut =
						Backend::HICANN::get_stdp_lut(*hicann_handle, synarray);

					if (stdp_lut != read_stdp_lut) {
						// disable whole synapsearray
						redman_hicann.synapsearrays()->disable(synarray, rewrite_policy);
					}
				}

				// check SYNRST register
				for (auto const& synarray : C::iter_all<C::SynapseArrayOnHICANN>()) {
					if (!redman_hicann_previous_test.synapsearrays()->has(synarray))
						continue;
					LOG4CXX_INFO(
					    test_logger, "Test SYNRST register on " << synarray
									    << " on HICANN " << hicann
									    << " with seed " << seed);

					HMF::HICANN::SynapseController::syn_rst_t syn_rst;

					syn_rst = number32bit(generator);

					// write and read values
					Backend::HICANN::set_syn_rst(*hicann_handle, synarray, syn_rst);
					HMF::HICANN::SynapseController::syn_rst_t read_syn_rst =
						Backend::HICANN::get_syn_rst(*hicann_handle, synarray);

					if (syn_rst != read_syn_rst) {
						// disable whole synapsearray
						redman_hicann.synapsearrays()->disable(synarray, rewrite_policy);
					}
				}

				// Reset synapse controller to default state to ensure functional settings in subsequent tests.
				for (auto const& synarray : C::iter_all<C::SynapseArrayOnHICANN>()) {
					if (!redman_hicann_previous_test.synapsearrays()->has(synarray))
						continue;
					Backend::HICANN::set_synapse_controller(*hicann_handle, synarray, synapse_controller);
				}

				// check fg ram values
				for (auto const& fg_block_c : C::iter_all<C::FGBlockOnHICANN>()) {
					// if component already blacklisted continue
					if (!redman_hicann_previous_test.fgblocks()->has(fg_block_c))
						continue;
					LOG4CXX_INFO(
					    test_logger, "Test fg ram values on " << fg_block_c << " on HICANN "
					                                          << hicann << " with seed " << seed);
					Backend::HICANN::FGRow fgr;
					fgr.setShared(number10bit(generator));
					for (auto const& nrn : C::iter_all<C::NeuronOnFGBlock>()) {
						fgr.setNeuron(nrn, number10bit(generator));
					}
					// write and read values
					Backend::HICANN::set_fg_ram_values(*hicann_handle, fg_block_c, fgr);
					Backend::HICANN::FGRow const read_fgr =
					    Backend::HICANN::get_fg_ram_values(*hicann_handle, fg_block_c);
					if (fgr != read_fgr) {
						redman_hicann.fgblocks()->disable(fg_block_c, rewrite_policy);
					}
				} // check fg ram values
			} // highspeed avail

			// check set_fg_config
			for (auto const& fg_block_c : C::iter_all<C::FGBlockOnHICANN>()) {
				// if component already blacklisted continue
				if (!redman_hicann_previous_test.fgblocks()->has(fg_block_c))
					continue;
				LOG4CXX_INFO(
				    test_logger, "Test component: " << fg_block_c << " on HICANN " << hicann
				                                    << " with seed " << seed);
				HMF::HICANN::FGConfig fgconfig;
				fgconfig.maxcycle = number8bit(generator);
				fgconfig.readtime = number6bit(generator);
				fgconfig.acceleratorstep = number6bit(generator);
				fgconfig.voltagewritetime = number6bit(generator);
				fgconfig.currentwritetime = number6bit(generator);
				fgconfig.fg_bias = number4bit(generator);
				fgconfig.fg_biasn = number4bit(generator);
				fgconfig.pulselength = number4bit(generator);
				fgconfig.groundvm = true_false(generator);
				fgconfig.calib = true_false(generator);

				// write and read values
				Backend::HICANN::set_fg_config(*hicann_handle, fg_block_c, fgconfig);
				HMF::HICANN::FGConfig const read_fgconfig =
				    Backend::HICANN::get_fg_config(*hicann_handle, fg_block_c);
				// disable defect fg_blocks
				if (fgconfig != read_fgconfig) {
					redman_hicann.fgblocks()->disable(fg_block_c, rewrite_policy);
				}
			} // check set_fg_config

			// check set_syndriver_switch_row
			for (auto const& syn_switch_row_c : C::iter_all<C::SynapseSwitchRowOnHICANN>()) {
				// if component already blacklisted continue
				if (!redman_hicann_previous_test.synapseswitchrows()->has(syn_switch_row_c))
					continue;
				LOG4CXX_INFO(
				    test_logger, "Test component: " << syn_switch_row_c << " on HICANN " << hicann
				                                    << " with seed " << seed);
				HMF::HICANN::SynapseSwitchRow syn_switch_row;
				// Fill with random values 4x4 bool
				for (auto const& row_number : C::iter_all<C::SynapseSwitchOnSynapseSwitchRow>()) {
					syn_switch_row[row_number.x().value()] = true_false(generator);
				}
				// write and read values
				Backend::HICANN::set_syndriver_switch_row(
				    *hicann_handle, syn_switch_row_c, syn_switch_row);
				HMF::HICANN::SynapseSwitchRow const read_syn_switch_row =
				    Backend::HICANN::get_syndriver_switch_row(*hicann_handle, syn_switch_row_c);
				// compare values
				for (auto const& row_number : C::iter_all<C::SynapseSwitchOnSynapseSwitchRow>()) {
					if (syn_switch_row[row_number.x().value()] != read_syn_switch_row[row_number.x().value()]) {
						// disable defect SynapseSwitch
						redman_hicann.synapseswitches()->disable(
						    C::SynapseSwitchOnHICANN(syn_switch_row_c, row_number), rewrite_policy);
					}
				}
			} // check set_syndriver_switch_row

			// check set_crossbar_switch_row
			for (auto const& hline : C::iter_all<C::HLineOnHICANN>()) {
				for (auto const& side : C::iter_all<C::Side>()) {
					LOG4CXX_INFO(
					    test_logger,
					    "Test component: CrossbarSwitchRow on " << side << ", " << hline
					    <<" on HICANN " << hicann << " with seed " << seed);
					HMF::HICANN::CrossbarRow crossbar_row;
					// Fill with random values 4x bool
					for (auto const& crossbar_switch : C::iter_all<C::CrossbarSwitchOnCrossbarSwitchRow>()) {
						crossbar_row[crossbar_switch.x().value()] = true_false(generator);
					}
					// write and read values
					Backend::HICANN::set_crossbar_switch_row(
					    *hicann_handle, hline, side, crossbar_row);
					HMF::HICANN::CrossbarRow const read_crossbar_row =
					    Backend::HICANN::get_crossbar_switch_row(*hicann_handle, hline, side);
					// compare values
					for (auto const& crossbar_switch : C::iter_all<C::CrossbarSwitchOnCrossbarSwitchRow>()) {
						if (crossbar_row[crossbar_switch.x().value()] != read_crossbar_row[crossbar_switch.x().value()]) {
							// disable defect crossbarswitches
							redman_hicann.crossbarswitches()->disable(
							    C::CrossbarSwitchOnHICANN(hline, side, crossbar_switch), rewrite_policy);
						}
					}
				}
			} // check set_crossbar_switch_row

		} catch (const std::exception& e) {
			LOG4CXX_ERROR(
			    test_logger, "Error during test of HICANN " << hicann << " using seed " << seed
			                                                << ": " << e.what());
			// reprogram HICANN
			::HMF::FPGA::init(fpga_handle, false);
			return EXIT_FAILURE;
		} catch (...) {
			LOG4CXX_ERROR(
			    test_logger, "Error during test of HICANN " << hicann << " using seed " << seed);
			// reprogram HICANN
			::HMF::FPGA::init(fpga_handle, false);
			return EXIT_FAILURE;
		}
	} // seed

	// check set_repeater with all possible settings
	// vertical
	try {
		for (auto const& vrepeater_c : C::iter_all<C::VRepeaterOnHICANN>()) {
			// if component already blacklisted continue
			if (!redman_hicann_previous_test.vrepeaters()->has(vrepeater_c))
				continue;
			LOG4CXX_INFO(test_logger, "Test component: " << vrepeater_c << " on HICANN " << hicann);
			test_repeater(hicann_handle, vrepeater_c, redman_hicann.vrepeaters(), rewrite_policy);
		} // vertical
		// horizontal
		for (auto const& hrepeater_c : C::iter_all<C::HRepeaterOnHICANN>()) {
			// if component already blacklisted continue
			if (!redman_hicann_previous_test.hrepeaters()->has(hrepeater_c))
				continue;
			LOG4CXX_INFO(test_logger, "Test component: " << hrepeater_c << " on HICANN " << hicann);
			test_repeater(hicann_handle, hrepeater_c, redman_hicann.hrepeaters(), rewrite_policy);
		} // horizontal
	} catch (const std::exception& e) {
		LOG4CXX_ERROR(
		    test_logger, "Error during repeater test on HICANN " << hicann << ": " << e.what());
		// reprogram HICANN
		::HMF::FPGA::init(fpga_handle, false);
		return EXIT_FAILURE;
	} catch (...) {
		LOG4CXX_ERROR(test_logger, "Error during repeater test on HICANN " << hicann);
		// reprogram HICANN
		::HMF::FPGA::init(fpga_handle, false);
		return EXIT_FAILURE;
	}

	// blacklist synapse array if it contains unstable synapses
	if (test_additional_components) {
		std::mt19937 generator(stability_seed);
		for (auto const& syn_array_c : C::iter_all<C::SynapseArrayOnHICANN>()) {
			try {
				LOG4CXX_INFO(
				    test_logger, "Test synapse stability on synapse array "
				                     << syn_array_c << " on HICANN " << hicann
				                     << " with fixed seed " << stability_seed);
				if (!test_synapse_array_stability(
				        syn_array_c, generator, hicann_handle, stability_repetitions,
				        synapse_controller)) {
					// disable synapsearray
					redman_hicann.synapsearrays()->disable(syn_array_c, rewrite_policy);
					LOG4CXX_INFO(
					    test_logger, "Synapse array " << syn_array_c << " on HICANN " << hicann
					                                  << " is unstable and was blacklisted");
				}
			} catch (...) {
				LOG4CXX_ERROR(
				    test_logger, "Error during array stability test of synapse array "
				                     << syn_array_c << " on HICANN " << hicann << " using seed "
				                     << stability_seed);
				// reprogram HICANN
				::HMF::FPGA::init(fpga_handle, false);
				return EXIT_FAILURE;
			}
		}
	}

	// store redman
	redman_hicann.save();

	// reprogram HICANN
	::HMF::FPGA::init(fpga_handle, false);
	LOG4CXX_INFO(test_logger, "Successfully tested HICANN " << hicann << " on Wafer " << wafer);

	return EXIT_SUCCESS;
}
