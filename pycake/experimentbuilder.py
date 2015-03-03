import pylogging
import Coordinate
import pyhalbe
import pysthal
import copy
from itertools import product
import numpy

from pycake.helpers.sthal import StHALContainer
from pycake.helpers.sthal import SimStHALContainer
from pycake.helpers.calibtic import Calibtic
from pycake.helpers.TraceAverager import createTraceAverager
from pycake.measure import ADCMeasurement
from pycake.measure import ADCMeasurementWithSpikes
from pycake.measure import SpikeMeasurement
from pycake.measure import I_gl_Measurement
from pycake.experiment import SequentialExperiment
import pycake.analyzer
from pycake.helpers.units import Unit

# shorter names
Enum = Coordinate.Enum
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter


class BaseExperimentBuilder(object):
    """ Builds a list of measurements from a config object.

        Args:
            config: pycake.config.Config object
            test:   (True|False) whether this should be a test measurement or not.
                    If it is a test measurement, the target parameter is also calibrated.
                    Otherwise, it stays uncalibrated even if there is calibration data for it.
    """
    logger = pylogging.get("pycake.experimentbuilder")

    def __init__(self, config, test=False):
        self.config = config

        path, name = self.config.get_calibtic_backend()
        wafer, hicann = self.config.get_coordinates()
        self.calibtic = Calibtic(path, wafer, hicann)

        self.neurons = self.config.get_neurons()
        self.blocks = self.config.get_blocks()
        self.test = test

    def generate_measurements(self):
        self.logger.INFO("Building experiment {}".format(self.config.get_target()))
        measurements = []
        coord_wafer, coord_hicann = self.config.get_coordinates()
        steps = self.config.get_steps()
        repetitions = self.config.get_repetitions()

        # Get readout shifts
        readout_shifts = self.get_readout_shifts(self.neurons)

        wafer_cfg = self.config.get_wafer_cfg()
        PLL = self.config.get_PLL()

        # Create one sthal container for each step
        # order is step 1, step 2, step 3, ..., step 1, step 2, step 3, ...
        for rep, step in product(range(repetitions), steps):
            self.logger.INFO(
                "Building step {}, repetition {}".format(step, rep))
            sim_denmem_cfg = self.config.get_sim_denmem()
            if sim_denmem_cfg:
                sthal = SimStHALContainer(
                    coord_wafer, coord_hicann, wafer_cfg=wafer_cfg, PLL=PLL,
                    config=self.config)
            else:
                sthal = StHALContainer(
                    coord_wafer, coord_hicann, wafer_cfg=wafer_cfg, PLL=PLL)
            # TODO maybe find better solution
            if self.test:
                step = step.copy()
                for value in step.itervalues():
                    if isinstance(value, Unit):
                        value.apply_calibration = True

            step_parameters = self.get_step_parameters(step)

            if not wafer_cfg:
                self.prepare_hicann(sthal.hicann, step_parameters)
                sthal.hicann.floating_gates = self.prepare_parameters(step_parameters)
            sthal = self.prepare_specific_config(sthal)

            measurement = self.make_measurement(sthal, self.neurons, readout_shifts)
            measurements.append(measurement)

        return measurements, repetitions

    def get_step_parameters(self, step):
        """ Get parameters for this step.
        """
        return self.config.get_step_parameters(step)

    def prepare_hicann(self, hicann, parameters):
        """Write HICANN configuration from parameters
        to sthal.hicann."""

        speedup_gl = parameters.get("speedup_gl", None)
        if speedup_gl is not None:
            hicann.set_speed_up_gl(speedup_gl)

        speedup_gladapt = parameters.get("speedup_gladapt", None)
        if speedup_gladapt is not None:
            hicann.set_speed_up_gladapt(speedup_gladapt)

        speedup_radapt = parameters.get("speedup_radapt", None)
        if speedup_radapt is not None:
            hicann.set_speed_up_radapt(speedup_radapt)

    def prepare_parameters(self, parameters):
        """ Writes floating gate parameters into a
        sthal FloatingGates container.
            This includes calibration and transformation from mV or nA to DAC values.

        Returns:
            pysthal.FloatingGates with given parameters
        """

        floating_gates = pysthal.FloatingGates()
        for neuron in self.neurons:
            neuron_params = copy.deepcopy(parameters)
            for param, value in neuron_params.iteritems():
                if (not isinstance(param, neuron_parameter)) or param.name[0] == '_':
                    # e.g. __last_neuron
                    continue

                value_dac = value.toDAC()

                if value.apply_calibration:
                    self.logger.TRACE("Applying calibration to coord {} value {}".format(neuron, value_dac))
                    value_dac.value = self.calibtic.apply_calibration(value_dac.value, param, neuron) + value_dac.offset

                self.logger.TRACE("Setting FGValue of {} parameter {} to {}.".format(neuron, param, value_dac))
                floating_gates.setNeuron(neuron, param, value_dac.value)

        for block in self.blocks:
            block_parameters = copy.deepcopy(parameters)
            for param, value in block_parameters.iteritems():
                if (not isinstance(param, shared_parameter)) or param.name[0] == '_':
                    # e.g. __last_*
                    continue
                # Check if parameter exists for this block
                even = block.id().value() % 2
                if even and param.name in ['V_clra', 'V_bout']:
                    continue
                if not even and param.name in ['V_clrc', 'V_bexp']:
                    continue

                value_dac = value.toDAC()

                # Do not calibrate target parameter except if this is a test measurement
                if value.apply_calibration:
                    self.logger.TRACE("Applying calibration to coord {} value {}".format(block, value_dac))
                    value_dac.value = self.calibtic.apply_calibration(value_dac.value, param, block) + value_dac.offset

                self.logger.TRACE("Setting FGValue of {} parameter {} to {}.".format(block, param, value_dac))
                floating_gates.setShared(block, param, value_dac.value)
        return floating_gates

    def prepare_specific_config(self, sthal):
        """ Hook function to specify additional stuff, e.g. current injection, spike generation, ...
        """
        return sthal

    def get_readout_shifts(self, neurons):
        """ Get readout shifts (in V) for a list of neurons.
            If no readout shift is saved, a shift of 0 is returned.

            Args:
                neurons: a list of NeuronOnHICANN coordinates
            Returns:
                shifts: a dictionary {neuron: shift (in V)}
        """
        return dict((n, self.calibtic.get_readout_shift(n)) for n in neurons)

    def get_experiment(self):
        """
        """
        measurements, repetitions = self.generate_measurements()
        analyzer = self.get_analyzer()
        experiment = SequentialExperiment(measurements, analyzer, repetitions)
        experiment = self.add_additional_measurements(experiment)
        return experiment
        return SequentialExperiment(measurements, analyzer, repetitions)

    def get_analyzer(self):
        """
        Get the appropriate analyzer for a specific parameter.
        Overwritte in subclass, when required.
        """
        return pycake.analyzer.MeanOfTraceAnalyzer()

    def make_measurement(self, sthal, neurons, readout_shifts):
        """
        Create a measurement object.
        Overwrite in subclass, when required.
        """

        return ADCMeasurement(sthal, neurons, readout_shifts)

    def add_additional_measurements(self, experiment):
        """ Dummy for adding additional initial measurements if needed.
        """
        return experiment

class InputSpike_Experimentbuilder(BaseExperimentBuilder):
    """Send input spikes to neurons"""
    def prepare_specific_config(self, sthal):
        spikes = self.config.get_input_spikes()
        sthal.set_input_spikes(spikes)
        sthal.stimulateNeurons(100000., 4)
        return sthal

    def make_measurement(self, sthal, neurons, readout_shifts):
        return ADCMeasurementWithSpikes(sthal, neurons, readout_shifts)


class E_l_Experimentbuilder(BaseExperimentBuilder):
    pass


class Spikes_Experimentbuilder(BaseExperimentBuilder):

    def get_analyzer(self):
        "get analyzer"
        return pycake.analyzer.Spikes_Analyzer()

    def make_measurement(self, sthal, neurons, readout_shifts):
        """
        Create a measurement object.
        Overwrite in subclass, when required.
        """

        return SpikeMeasurement(sthal, neurons)


class V_reset_Experimentbuilder(BaseExperimentBuilder):
    def get_analyzer(self):
        "get analyzer"
        return pycake.analyzer.V_reset_Analyzer()

    def prepare_specific_config(self, sthal):
        sthal.set_recording_time(25.0e-6, 4)
        sthal.maximum_spikes = 3
        return sthal


class V_t_Experimentbuilder(BaseExperimentBuilder):
    def get_analyzer(self):
        "get analyzer"
        return pycake.analyzer.V_t_Analyzer()

    def prepare_specific_config(self, sthal):
        sthal.set_recording_time(25.0e-6, 4)
        sthal.maximum_spikes = 3
        return sthal


class I_gladapt_Experimentbuilder(BaseExperimentBuilder):
    def get_analyzer(self):
        """ISI std is a hint whether adaptation takes place.
        Should be replaced by a better analysis before real usage.
        """
        return pycake.analyzer.ISI_Analyzer()


class I_pl_Experimentbuilder(BaseExperimentBuilder):
    """Longer recording time than parent class.

    To be used with hardware."""
    def prepare_specific_config(self, sthal):
        sthal.recording_time = 1e-3
        sthal.maximum_spikes = 10
        return sthal

    def get_analyzer(self):
        "get analyzer"
        return pycake.analyzer.ISI_Analyzer()


class E_synx_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        sthal.simulation_init_time = 100.0e-6
        sthal.set_recording_time(10e-6, 10)
        sthal.stimulateNeurons(5.0e6, 1, excitatory=True)
        sthal.maximum_spikes = 1
        sthal.spike_counter_offset = 0.0
        return sthal


class E_syni_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        sthal.simulation_init_time = 100.0e-6
        sthal.set_recording_time(10e-6, 10)
        sthal.stimulateNeurons(5.0e6, 1, excitatory=False)
        sthal.maximum_spikes = 1
        sthal.spike_counter_offset = 0.0
        return sthal


class V_convoff_Experimentbuilder(BaseExperimentBuilder):
    EXCITATORY = None
    ANALYZER = None

    def __init__(self, *args, **kwargs):
        super(V_convoff_Experimentbuilder, self).__init__(*args, **kwargs)
        self.init_time = 0.0e-6
        self.recording_time = 360e-6
        self.spikes = numpy.array([320e-6])
        self.gmax_div = 30

    def prepare_specific_config(self, sthal):
        sthal.simulation_init_time = self.init_time
        sthal.set_recording_time(self.recording_time, 1)
        sthal.send_spikes_to_all_neurons(
            self.spikes, excitatory=self.EXCITATORY, gmax_div=self.gmax_div)
        sthal.maximum_spikes = 2
        sthal.spike_counter_offset = 0.0
        return sthal

    def make_measurement(self, sthal, neurons, readout_shifts):
        return ADCMeasurementWithSpikes(sthal, neurons, readout_shifts)

    def get_analyzer(self):
        "get analyzer"
        return self.ANALYZER(self.spikes)


class V_convoffi_Experimentbuilder(V_convoff_Experimentbuilder):
    EXCITATORY = False
    ANALYZER = pycake.analyzer.V_convoffi_Analyzer

    def __init__(self, *args, **kwargs):
        super(V_convoffi_Experimentbuilder, self).__init__(*args, **kwargs)
        self.recording_time = 80e-6
        self.spikes = numpy.array([50e-6])


class V_convoffx_Experimentbuilder(V_convoff_Experimentbuilder):
    EXCITATORY = True
    ANALYZER = pycake.analyzer.V_convoffx_Analyzer

class V_convoffi_Experimentbuilder(BaseExperimentBuilder):
    EXCITATORY = None
    ANALYZER = pycake.analyzer.V_convoffx_Analyzer

from calibration.vsyntc import SimplePSPAnalyzer

class V_syntcx_Experimentbuilder(V_convoff_Experimentbuilder):
    EXCITATORY = True
    ANALYZER = SimplePSPAnalyzer

class V_syntci_Experimentbuilder(V_convoff_Experimentbuilder):
    EXCITATORY = False
    ANALYZER = SimplePSPAnalyzer

class I_gl_Experimentbuilder(BaseExperimentBuilder):
    def __init__(self, *args, **kwargs):
        super(I_gl_Experimentbuilder, self).__init__(*args, **kwargs)

    def prepare_specific_config(self, sthal):
        sthal.recording_time = 5e-3
        return sthal

    def make_measurement(self, sthal, neurons, readout_shifts):
        return I_gl_Measurement(sthal, neurons, readout_shifts)

    def get_analyzer(self):
        """ Get the appropriate analyzer for a specific parameter.
        """
        coord_wafer, coord_hicann = self.config.get_coordinates()
        # not ideal place to call createTraceAverager, connects to hardware
        trace_averager = createTraceAverager(coord_wafer, coord_hicann)
        save_traces = self.config.get_save_traces()
        pll = self.config.get_PLL()
        return pycake.analyzer.I_gl_Analyzer(trace_averager, save_traces, pll)


class E_l_I_gl_fixed_Experimentbuilder(E_l_Experimentbuilder):
    def get_analyzer(self):
        """ Get the appropriate analyzer for a specific parameter.
        """
        return pycake.analyzer.MeanOfTraceAnalyzer()


#from calibration.vsyntc import V_syntci_Experimentbuilder
#from calibration.vsyntc import V_syntcx_Experimentbuilder


class V_syntci_psp_max_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        sthal.stimulateNeurons(0.1e6, 1, excitatory=False)
        return sthal


class V_syntcx_psp_max_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        sthal.stimulateNeurons(0.1e6, 1, excitatory=True)
        return sthal


class readout_shift_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        sthal.set_neuron_size(64)
        return sthal

    def get_readout_shifts(self, neurons):
        """ Get readout shifts (in V) for a list of neurons.
            If no readout shift is saved, a shift of 0 is returned.

            Args:
                neurons: a list of NeuronOnHICANN coordinates
            Returns:
                shifts: a dictionary {neuron: shift (in V)}
        """
        return dict((n, 0) for n in neurons)


class I_bexp_Experimentbuilder(BaseExperimentBuilder):
    def get_analyzer(self):
        return pycake.analyzer.ISI_Analyzer()
