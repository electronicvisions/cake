import numpy as np
import pylogging
import Coordinate
import pyhalbe
import pysthal
import copy
from itertools import product

from pycake.helpers.units import Current, Voltage, DAC
from pycake.helpers.sthal import StHALContainer
from pycake.helpers.calibtic import Calibtic
from pycake.measure import Measurement, I_gl_Measurement
from pycake.experiment import SequentialExperiment
import pycake.analyzer

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

    def __init__(self, config, test=False, trace_readout_enabled=True, spike_readout_enabled=False):
        self.config = config

        path, name = self.config.get_calibtic_backend()
        wafer, hicann = self.config.get_coordinates()
        self.calibtic = Calibtic(path, wafer, hicann)

        self.neurons = self.config.get_neurons()
        self.blocks = self.config.get_blocks()
        self.test = test

        self.trace_readout_enabled = trace_readout_enabled
        self.spike_readout_enabled = spike_readout_enabled

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
            self.logger.INFO("Building step {}, repetition {}".format(step,rep))
            sthal = StHALContainer(coord_wafer, coord_hicann, wafer_cfg=wafer_cfg, PLL=PLL)
            # TODO maybe find better solution
            if self.test:
                step = step.copy()
                for value in step.itervalues():
                    value.apply_calibration = True

            step_parameters = self.get_step_parameters(step)

            if not wafer_cfg:
                sthal.hicann.floating_gates = self.prepare_parameters(step_parameters)
            sthal = self.prepare_specific_config(sthal)

            measurement = self.make_measurement(sthal, self.neurons, readout_shifts)
            measurements.append(measurement)

        return measurements, repetitions

    def get_step_parameters(self, step):
        """ Get parameters for this step.
        """
        return self.config.get_step_parameters(step)

    def prepare_parameters(self, parameters):
        """ Writes parameters into a sthal container.
            This includes calibration and transformation from mV or nA to DAC values.

        Returns:
            pysthal.FloatingGates with given parameters
        """

        floating_gates = pysthal.FloatingGates()
        for neuron in self.neurons:
            neuron_params = copy.deepcopy(parameters)
            for param, value in neuron_params.iteritems():
                if isinstance(param, shared_parameter) or param.name[0] == '_':
                    continue

                value_dac = value.toDAC()

                if value.apply_calibration:
                    self.logger.TRACE("Applying calibration to coord {} value {}".format(neuron, value_dac))
                    value_dac.value = self.calibtic.apply_calibration(value_dac.value, param, neuron)

                self.logger.TRACE("Setting FGValue of {} parameter {} to {}.".format(neuron, param, value_dac))
                floating_gates.setNeuron(neuron, param, value_dac.value)

        for block in self.blocks:
            block_parameters = copy.deepcopy(parameters)
            for param, value in block_parameters.iteritems():
                if isinstance(param, neuron_parameter) or param.name[0] == '_':
                    continue
                # Check if parameter exists for this block
                even = block.id().value()%2
                if even and param.name in ['V_clra', 'V_bout']:
                    continue
                if not even and param.name in ['V_clrc', 'V_bexp']:
                    continue

                value_dac = value.toDAC()

                # Do not calibrate target parameter except if this is a test measurement
                if value.apply_calibration:
                    self.logger.TRACE("Applying calibration to coord {} value {}".format(block, value_dac))
                    value_dac.value = self.calibtic.apply_calibration(value_dac.value, param, block)

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

        return Measurement(sthal, neurons, readout_shifts,
                           self.trace_readout_enabled,
                           self.spike_readout_enabled)

class E_l_Experimentbuilder(BaseExperimentBuilder):
    pass

class Spikes_Experimentbuilder(BaseExperimentBuilder):
    def __init__(self, *args, **kwargs):
        super(Spikes_Experimentbuilder, self).__init__(*args, **kwargs)

        self.trace_readout_enabled = False
        self.spike_readout_enabled = True

class V_reset_Experimentbuilder(BaseExperimentBuilder):
    def get_readout_shifts(self, neurons):
        if self.test:
            return super(V_reset_Experimentbuilder, self).get_readout_shifts(
                    neurons)
        else:
           return dict(((n, 0) for n in neurons))

    def get_analyzer(self):
        "get analyzer"
        return pycake.analyzer.V_reset_Analyzer()

class V_t_Experimentbuilder(BaseExperimentBuilder):
    def get_analyzer(self):
        "get analyzer"
        return pycake.analyzer.V_t_Analyzer()

class I_pl_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        sthal.recording_time = 1e-3
        return sthal

    def get_analyzer(self):
        "get analyzer"
        return pycake.analyzer.I_pl_Analyzer()

class E_syni_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        sthal.stimulateNeurons(5.0e6, 1, excitatory=False)
        return sthal

class E_synx_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        sthal.stimulateNeurons(5.0e6, 1, excitatory=True)
        return sthal

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
        c_w, c_h = self.config.get_coordinates()
        save_traces = self.config.get_save_traces()
        return pycake.analyzer.I_gl_Analyzer(c_w, c_h, save_traces)

class E_l_I_gl_fixed_Experimentbuilder(E_l_Experimentbuilder):
    def get_analyzer(self):
        """ Get the appropriate analyzer for a specific parameter.
        """
        return pycake.analyzer.MeanOfTraceAnalyzer()

from calibration.vsyntc import V_syntci_Experimentbuilder
from calibration.vsyntc import V_syntcx_Experimentbuilder

class V_syntci_psp_max_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        sthal.stimulateNeurons(0.1e6, 1, excitatory=False)
        return sthal

class V_syntcx_psp_max_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        sthal.stimulateNeurons(0.1e6, 1, excitatory=True)
        return sthal
