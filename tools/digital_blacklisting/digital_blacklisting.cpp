#include <iostream>
#include <random>
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

// sets has_value = True if component not already touched
template <typename res>
void touch_component(boost::shared_ptr<res> component)
{
	if (!component->has_value())
		component->enable_all();
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
	    "path to blacklisting data")(
	    "output_backend_path", po::value<std::string>(&output_backend_path)->default_value("./"),
	    "path where blacklisting results of this test are stored")(
	    "highspeed", po::value<bool>(&highspeed)->default_value(1), "use highspeed otherwise JTAG")(
	    "seeds", po::value<std::vector<size_t> >(&seeds)->multitoken()->required(), "used seeds");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, options), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cout << options << "\n";
		return EXIT_SUCCESS;
	}

	logger_default_config(log4cxx::Level::getInfo());
	log4cxx::LoggerPtr test_logger = log4cxx::Logger::getLogger("cake.digital_blacklisting");

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
	input_backend->config("path", input_backend_path);
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
	// stores if currently used hicanns has highspeed -> used to skip some tests without highspeed
	bool highspeed_avail = false;
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
				highspeed_avail = true;
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
	HMF::FPGA::init(fpga_handle, false);

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
	std::uniform_int_distribution<int> number9(0, 9);
	std::uniform_int_distribution<int> number15(0, 15);
	std::uniform_int_distribution<int> number223(0, 223);
	std::uniform_int_distribution<int> number4bit(0, 15);
	std::uniform_int_distribution<int> number6bit(0, 63);
	std::uniform_int_distribution<int> number8bit(0, 255);
	std::uniform_int_distribution<int> number16bit(0, 65535);
	std::uniform_int_distribution<int> numberRepBlock(
	    0, (1 << halco::hicann::v2::TestPortOnRepeaterBlock::end) - 1);
	std::uniform_int_distribution<int> numberstpcap(
	    0, (1 << HMF::HICANN::SynapseDriver::num_cap) - 1);

	// Some coordinates are not available on HICANN v4
	std::vector<C::SynapseDriverOnHICANN> unavailable_syn_drv;
	for (size_t drv = 110; drv < 114; drv++) {
		unavailable_syn_drv.push_back(C::SynapseDriverOnHICANN(C::Enum(drv)));
	}
	std::vector<C::SynapseRowOnHICANN> unavailable_syn_row;
	for (size_t row = 220; row < 228; row++) {
		unavailable_syn_row.push_back(C::SynapseRowOnHICANN(row));
	}

	// Once touch all tested components to set has_value = True
	// Used to distinguish between tested and not tested components
	// If error occurs during test xml file is not saved anyway
	if (highspeed_avail) {
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
			if (highspeed_avail) {
				// check set_Synapse_Driver
				// iterate Hicann Synapse Driver
				for (auto syn_drv_c : C::iter_all<C::SynapseDriverOnHICANN>()) {
					if (std::find(unavailable_syn_drv.begin(), unavailable_syn_drv.end(), syn_drv_c) !=
					    unavailable_syn_drv.end()) {
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
					Backend::HICANN::set_synapse_driver(*hicann_handle, syn_drv_c, syn_drv);
					HMF::HICANN::SynapseDriver const read_synapse_driver =
					    Backend::HICANN::get_synapse_driver(*hicann_handle, syn_drv_c);
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
					if (std::find(unavailable_syn_row.begin(), unavailable_syn_row.end(), syn_row_c) !=
					    unavailable_syn_row.end()) {
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
					Backend::HICANN::set_weights_row(*hicann_handle, syn_row_c, weight_row);
					HMF::HICANN::WeightRow const read_weight_row =
					    Backend::HICANN::get_weights_row(*hicann_handle, syn_row_c);

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
					if (std::find(unavailable_syn_drv.begin(), unavailable_syn_drv.end(), syn_drv_c) !=
					    unavailable_syn_drv.end()) {
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
					    *hicann_handle, syn_drv_c, decoder_double_row);
					HMF::HICANN::DecoderDoubleRow const read_decoder_row =
					    Backend::HICANN::get_decoder_double_row(*hicann_handle, syn_drv_c);
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

				// check set_dnc_merger
				HMF::HICANN::DNCMergerLine merger_line;
				for (auto const& merger_c : C::iter_all<C::DNCMergerOnHICANN>()) {
					// possible values of Merger: RIGHT_ONLY=0, LEFT_ONLY=1, MERGE=2 or use
					// HMF::Merge::MERGE
					HMF::HICANN::DNCMerger merger(number2(generator), true_false(generator));
					merger_line[merger_c] = merger;
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

				// check set_merger_tree
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

			// check set_repeater
			// overloaded with horizontal and vertical repeater
			// vertical
			for (auto const& vrepeater_c : C::iter_all<C::VRepeaterOnHICANN>()) {
				// if component already blacklisted continue
				if (!redman_hicann_previous_test.vrepeaters()->has(vrepeater_c))
					continue;
				LOG4CXX_INFO(
				    test_logger, "Test component: " << vrepeater_c << " on HICANN " << hicann
				                                    << " with seed " << seed);
				HMF::HICANN::VerticalRepeater vrepeater;
				vrepeater.setRen(number3(generator));
				vrepeater.setLen(number3(generator));
				switch (number4(generator)) {
					case 0:
						vrepeater.setIdle();
						break;
					case 1:
						vrepeater.setForwarding(C::SideVertical(true_false(generator)));
						break;
					case 2:
						vrepeater.setInput(C::SideVertical(true_false(generator)));
						break;
					// set output: both directions false not allowed
					case 3:
						switch (number2(generator)) {
							case 0:
								vrepeater.setOutput(C::SideVertical(0), true);
								vrepeater.setOutput(C::SideVertical(1), false);
								break;
							case 1:
								vrepeater.setOutput(C::SideVertical(1), true);
								vrepeater.setOutput(C::SideVertical(0), false);
								break;
							case 2:
								vrepeater.setOutput(C::SideVertical(0), true);
								vrepeater.setOutput(C::SideVertical(1), true);
								break;
						}
						break;
					case 4:
						vrepeater.setLoopback();
						break;
				}
				// write and read values
				Backend::HICANN::set_repeater(*hicann_handle, vrepeater_c, vrepeater);
				HMF::HICANN::VerticalRepeater const read_vrepeater =
				    Backend::HICANN::get_repeater(*hicann_handle, vrepeater_c);
				// disable defect vrepeater
				if (vrepeater != read_vrepeater) {
					redman_hicann.vrepeaters()->disable(vrepeater_c, rewrite_policy);
				}
			} // check vertical

			// horizontal
			for (auto const& hrepeater_c : C::iter_all<C::HRepeaterOnHICANN>()) {
				// if component already blacklisted continue
				if (!redman_hicann_previous_test.hrepeaters()->has(hrepeater_c))
					continue;
				LOG4CXX_INFO(
				    test_logger, "Test component: " << hrepeater_c << " on HICANN " << hicann
				                                    << " with seed " << seed);
				HMF::HICANN::HorizontalRepeater hrepeater;
				hrepeater.setRen(number3(generator));
				hrepeater.setLen(number3(generator));
				switch (number4(generator)) {
					case 0:
						hrepeater.setIdle();
						break;
					case 1:
						hrepeater.setForwarding(C::SideHorizontal(true_false(generator)));
						break;
					case 2:
						hrepeater.setInput(C::SideHorizontal(true_false(generator)));
						break;
					// set output: not allowed that both directions are false
					case 3:
						switch (number2(generator)) {
							case 0:
								hrepeater.setOutput(C::SideHorizontal(0), true);
								hrepeater.setOutput(C::SideHorizontal(1), false);
								break;
							case 1:
								hrepeater.setOutput(C::SideHorizontal(1), true);
								hrepeater.setOutput(C::SideHorizontal(0), false);
								break;
							case 2:
								hrepeater.setOutput(C::SideHorizontal(0), true);
								hrepeater.setOutput(C::SideHorizontal(1), true);
								break;
						}
						break;
					case 4:
						hrepeater.setLoopback();
						break;
				}
				// write and read values
				Backend::HICANN::set_repeater(*hicann_handle, hrepeater_c, hrepeater);
				HMF::HICANN::HorizontalRepeater const read_hrepeater =
				    Backend::HICANN::get_repeater(*hicann_handle, hrepeater_c);
				// disable defect vrepeater
				if (hrepeater != read_hrepeater) {
					redman_hicann.hrepeaters()->disable(hrepeater_c, rewrite_policy);
				}
			} // check horizontal

		} catch (const std::exception& e) {
			LOG4CXX_ERROR(
			    test_logger, "Error during test of HICANN " << hicann << " using seed " << seed
			                                                << ": " << e.what());
			// reset HICANN
			::HMF::FPGA::init(fpga_handle, false);
			return EXIT_FAILURE;
		} catch (...) {
			LOG4CXX_ERROR(
			    test_logger, "Error during test of HICANN " << hicann << " using seed " << seed);
			// reset HICANN
			::HMF::FPGA::init(fpga_handle, false);
			return EXIT_FAILURE;
		}
	} // seed

	// store redman
	redman_hicann.save();

	// reset HICANN
	::HMF::FPGA::init(fpga_handle, false);
	LOG4CXX_INFO(test_logger, "Successfully tested HICANN " << hicann << " on Wafer " << wafer);

	return EXIT_SUCCESS;
}
