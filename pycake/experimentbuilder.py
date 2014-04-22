import numpy as np
import pylogging
import pyhalbe
import pycalibtic
import time

import pyredman as redman

from pycake.helpers.redman import init_backend as init_redman
from pycake.helpers.units import Current, Voltage, DAC
from pycake.helpers.sthal import StHALContainer
from pycake.measure import Measurement, I_gl_Measurement
import pycake.analyzer

# shorter names
Coordinate = pyhalbe.Coordinate
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
    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['logger']
        return odict

    def __setstate__(self, dic):
        dic['logger'] = pylogging.get("pycake.experimentbuilder")
        self.__dict__.update(dic)

    def __init__(self, config, test=False):
        self.config = config

        name, path = self.config.get_calibtic_backend()
        wafer, hicann = self.config.get_coordinates()
        self.calibtic = pycake.helpers.calibtic.Calibtic(path, wafer, hicann)

        self.neurons = self.config.get_neurons()
        self.blocks = self.config.get_blocks()
        self.target_parameter = self.config.target_parameter
        self.test = test
        self.logger = pylogging.get("pycake.experimentbuilder")

    def generate_measurements(self):
        self.logger.INFO("{} - Building experiment for parameter {}".format(time.asctime(), self.target_parameter.name))
        config = self.config
        measurements = []
        coord_wafer, coord_hicann = self.config.get_coordinates()
        steps = self.config.get_steps()
        parameters = self.config.get_parameters()
        target_parameter = self.target_parameter

        # Initialize calibtic backend
        cal_path, cal_name = self.config.get_calibtic_backend()
        hc, nc, bc, md = self.calibtic.load_calibration()

        # Get readout shifts
        readout_shifts = self.get_readout_shifts(self.neurons, nc)

        # Create one sthal container for each step
        for step in steps:
            sthal = StHALContainer(coord_wafer, coord_hicann)
            step_parameters = self.get_step_parameters(config, step)
            sthal = self.prepare_parameters(sthal, step_parameters, nc, bc)
            sthal = self.prepare_specific_config(sthal)

            measurement = self.make_measurement(sthal, self.neurons, readout_shifts)
            measurements.append(measurement)

        return measurements

    def get_step_parameters(self, config, step):
        """ Get parameters for this step.
        """
        return config.get_step_parameters(step)

    def make_measurement(self, sthal, neurons, readout_shifts):
        return Measurement(sthal, neurons, readout_shifts)

    def prepare_parameters(self, sthal, parameters, nc, bc):
        """ Writes parameters into a sthal container.
            This includes calibration and transformation from mV or nA to DAC values.
        """
        fgc = pyhalbe.HICANN.FGControl()
        self.logger.INFO("Preparing parameters")

        for neuron in self.neurons:
            neuron_id = neuron.id().value()
            if neuron_id == 1: # Only give this info for some neurons to avoid spamming cout
                self.logger.TRACE("Preparing parameters for neuron {}".format(neuron_id))
            for param, value in parameters.iteritems():
                name = param.name
                if isinstance(param, shared_parameter) or name[0] == '_':
                    continue

                ## TODO remove this --> only debug lines
                #if param == self.target_parameter:
                #    value.apply_calibration = True
                #######################################

                # Do not calibrate target parameter except if this is a test measurement
                if value.apply_calibration and ((not param == self.target_parameter) or self.test):
                    try:
                        calibrated = nc.at(neuron_id).at(param).apply(value.value)
                        if neuron_id == 1: # Only give this info for some neurons to avoid spamming cout
                            self.logger.TRACE("Calibrated neuron {} parameter {} to value {}".format(neuron_id, param.name, calibrated))
                    except:
                        # TODO proper implementation (give warning etc.)
                        calibrated = value.value
                        if neuron_id == 1: # Only give this info for some neurons to avoid spamming cout
                            self.logger.TRACE("No calibration found for parameter {}.".format(param.name))
                else:
                    calibrated = value.value
                    if neuron_id == 1: # Only give this info for some neurons to avoid spamming cout
                        self.logger.TRACE("No calibration wanted for parameter {}.".format(param.name))

                calibrated = self.check_range(calibrated, param)
                value.value = calibrated
                int_value = int(round(value.toDAC().value))
                if neuron_id == 1:
                    self.logger.TRACE("Setting neuron {} parameter {} to value {}".format(neuron_id, param.name, int_value))
                fgc.setNeuron(neuron, param, int_value)

        for block in self.blocks:
            block_id = block.id().value()
            for param, value in parameters.iteritems():
                name = param.name
                if isinstance(param, neuron_parameter) or name[0] == '_':
                    continue
                # Check if parameter exists for this block
                even = block_id%2
                if even and name in ['V_clrc', 'V_bout']:
                    continue
                if not even and name in ['V_clra', 'V_bexp']:
                    continue

                # Do not calibrate target parameter except if this is a test measurement
                if value.apply_calibration and ((not param == self.target_parameter) or self.test):
                    try:
                        calibrated = bc.at(neuron_id).at(param).apply(value.value)
                        #self.logger.TRACE("Calibrated block {} parameter {} to value {}".format(block_id, param.name, calibrated))
                    except:
                        # TODO proper implementation (give warning etc.)
                        calibrated = value.value
                        #self.logger.TRACE("No calibration found for parameter {}.".format(param.name))
                else:
                    calibrated = value.value
                    #self.logger.TRACE("No calibration wanted for parameter {}.".format(param.name))
                value.value = calibrated
                int_value = int(round(value.toDAC().value))
                if block_id == 1:
                    self.logger.TRACE("Setting block {} parameter {} to value {}".format(block_id, param.name, int_value))
                fgc.setShared(block, param, int_value)

        sthal.hicann.floating_gates = fgc
        return sthal

    def check_range(self, value, parameter):
        if parameter.name[0] == 'I':
            upperbound = 2500
        else:
            upperbound = 1800
        if value < 0:
            self.logger.TRACE("Value lower than 0. Clipping")
            return 0
        elif value > upperbound:
            self.logger.TRACE("Value higher than 1023. Clipping")
            return upperbound
        else:
            return value 

    def prepare_specific_config(self, sthal):
        """ Hook function to specify additional stuff, e.g. current injection, spike generation, ...
        """
        return sthal

    def get_readout_shifts(self, neurons, nc):
        """ Get readout shifts (in V) for a list of neurons.
            If no readout shift is saved, a shift of 0 is returned.

            Args:
                neurons: a list of NeuronOnHICANN coordinates
            Returns:
                shifts: a dictionary {neuron: shift (in V)}
        """
        if not isinstance(neurons, list):
            neurons = [neurons]
        shifts = {}
        for neuron in neurons:
            neuron_id = neuron.id().value()
            try:
                # Since readout shift is a constant, return the value for DAC = 0
                shift = nc.at(neuron_id).at(21).apply(0) # Convert to mV
                shifts[neuron] = shift
            except:
                #self.logger.WARN("No readout shift calibration for neuron {} found. Using unshifted values.".format(neuron))
                shifts[neuron] = 0
        return shifts

    def get_analyzer(self, parameter): # TODO move to experimentbuilder
        """ Get the appropriate analyzer for a specific parameter.
        """
        if parameter == neuron_parameter.I_gl:
            c_w, c_h = self.config.get_coordinates()
            return pycake.analyzer.I_gl_Analyzer(c_w, c_h)
        else:
            AnalyzerType = getattr(pycake.analyzer, "{}_Analyzer".format(parameter.name))
            return AnalyzerType()

    # TODO implement redman
    #def init_redman(self, backend):
    #    """Initialize defect management for given backend."""
    #    # FIXME default coordinates
    #    coord_hglobal = self.sthal.hicann.index()  # grab HICANNGlobal from StHAL
    #    coord_wafer = coord_hglobal.wafer()
    #    coord_hicann = coord_hglobal.on_wafer()
    #    wafer = redman.Wafer(backend, coord_wafer)
    #    if not wafer.hicanns().has(coord_hicann):
    #        raise ValueError("HICANN {} is marked as defect.".format(int(coord_hicann.id())))
    #    hicann = wafer.find(coord_hicann)
    #    self._red_hicann = hicann
    #    self._red_nrns = hicann.neurons()


class E_l_Experimentbuilder(BaseExperimentBuilder):
    def get_step_parameters(self, config, step):
        """ For E_l, the reversal potentials need to be set appropriately
        """
        parameters =  config.get_step_parameters(step)
        dist = config.get_E_syn_dist()
        parameters[neuron_parameter.E_syni] = Voltage(step + dist['E_syni'])
        parameters[neuron_parameter.E_synx] = Voltage(step + dist['E_synx'])
        return parameters

class V_reset_Experimentbuilder(BaseExperimentBuilder):
    pass

class V_t_Experimentbuilder(BaseExperimentBuilder):
    pass

class E_syni_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        sthal.stimulateNeurons(5.0e6, 4)
        return sthal

class E_synx_Experimentbuilder(E_syni_Experimentbuilder):
    pass

class I_gl_Experimentbuilder(BaseExperimentBuilder):
    def prepare_specific_config(self, sthal):
        """ Prepares current stimulus and increases recording time.
        """
        sthal.recording_time = 5e-3
        pulse_length = 15
        stim_current = 35
        stim_length = 65

        stimulus = pyhalbe.HICANN.FGStimulus()
        stimulus.setPulselength(pulse_length)
        stimulus.setContinuous(True)

        stimulus[:stim_length] = [stim_current] * stim_length
        stimulus[stim_length:] = [0] * (len(stimulus) - stim_length)

        sthal.set_current_stimulus(stimulus)
        return sthal

    def make_measurement(self, sthal, neurons, readout_shifts):
        return I_gl_Measurement(sthal, neurons, readout_shifts)

class V_syntc_Experimentbuilder(BaseExperimentBuilder):
    pass # TODO by CK

