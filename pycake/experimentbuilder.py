import numpy as np
import pylogging
import pyhalbe
import pycalibtic
from pyhalbe.Coordinate import NeuronOnHICANN, FGBlockOnHICANN
from pycake.helpers.calibtic import init_backend as init_calibtic
from pycake.helpers.redman import init_backend as init_redman
import pyredman as redman
from pycake.helpers.units import Current, Voltage, DAC
import pycake.helpers.misc as misc
from pycake.helpers.sthal import StHALContainer
from pycake.measure import Measurement

# shorter names
Coordinate = pyhalbe.Coordinate
Enum = Coordinate.Enum
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

class BaseExperimentBuilder(object):
    """ Builds a list of measurements from a config object.
        This config object knows its target parameter etc.
    """
    def __init__(self, config):
        self.config = config
        self.neurons = [Coordinate.NeuronOnHICANN(Enum(i)) for i in range(512)] # TODO fix this
        self.blocks = [Coordinate.FGBlockOnHICANN(Enum(i)) for i in range(4)]
        self.target_parameter = self.config.target_parameter

    def generate(self, config):
        measurements = []
        coord_wafer, coord_hicann = self.config.get_coordinates()
        steps = self.config.get_steps()
        parameters = self.config.get_parameters()
        target_parameter = self.target_parameter

        # Initialize calibtic backend
        cal_path, cal_name = self.config.get_calibtic_backend()
        hc, nc, bc, md = self.load_calibration(cal_path, cal_name)

        # Get readout shifts
        readout_shifts = self.get_readout_shifts(self.neurons, nc)

        # Create one sthal container for each step
        for step in steps:
            sthal = StHALContainer(coord_wafer, coord_hicann)
            step_parameters = self.get_step_parameters(parameters, step, target_parameter)
            sthal = self.prepare_parameters(sthal, step_parameters, nc, bc)

            measurement = Measurement(sthal, self.neurons, readout_shifts)
            measurements.append(measurement)

        return measurements

    def load_calibration(self, path, name):
        """Initialize Calibtic backend, load existing calibration data."""
        calibtic_backend = init_calibtic(type='xml', path=path)

        hc = pycalibtic.HICANNCollection()
        nc = hc.atNeuronCollection()
        bc = hc.atBlockCollection()
        md = pycalibtic.MetaData()

        # Delete all standard entries. TODO: fix calibtic to use proper standard entries
        for nid in range(512):
            nc.erase(nid)
        for bid in range(4):
            bc.erase(bid)

        try:
            calibtic_backend.load(name, md, hc)
            # load existing calibration:
            nc = hc.atNeuronCollection()
            bc = hc.atBlockCollection()
        except RuntimeError, e:
            if e.message != "data set not found":
                raise RuntimeError(e)
            else:
                # backend does not exist
                pass

        return (hc, nc, bc, md)

    def get_calibrated(self, coord, value, parameter, nc, bc):
        #TODO add calibration logic
        return value

    def get_step_parameters(self, parameters, step, target_parameter):
        """ Takes a parameter dict and updates only the entry that defines the step
        """
        if target_parameter.name[0] == "I":
            unit = Current
        else:
            unit = Voltage
        parameters[target_parameter] = unit(step)
        return parameters

    def prepare_parameters(self, sthal, parameters, nc, bc):
        """ Prepares parameters on a sthal container.
            This includes calibration and transformation to DAC values.
        """
        fgc = pyhalbe.HICANN.FGControl()

        for neuron in self.neurons:
            neuron_id = neuron.id().value()
            for param, value in parameters.iteritems():
                name = param.name
                if isinstance(param, shared_parameter) or name[0] == '_':
                    continue
                dac_value = value.toDAC().value
                if value.apply_calibration: 
                    try:
                        calibrated_dac = nc.at(neuron_id).at(param).apply(dac_value)
                    except:
                        # TODO proper implementation (give warning etc.)
                        calibrated_dac = dac_value
                else:
                    calibrated_dac = dac_value
                int_value = int(round(calibrated_dac))
                fgc.setNeuron(neuron, param, int_value)

        for block in self.blocks:
            block_id = block.id().value()
            for param, value in parameters.iteritems():
                name = param.name
                if isinstance(param, neuron_parameter) or name[0] == '_':
                    continue
                even = block_id%2
                if even and (name == 'V_clrc' or 'V_bout'):
                    continue
                if not even and (name == 'V_clra' or 'V_bexp'):
                    continue
                dac = value.toDAC()
                if value.apply_calibration: 
                    try:
                        calibrated_dac = bc.at(neuron_id).at(param).apply(dac_value)
                    except:
                        # TODO proper implementation (give warning etc.)
                        calibrated_dac = dac_value
                else:
                    calibrated_dac = dac_value
                int_value = int(round(calibrated_dac))
                fgc.setShared(block, param, int_value)

        sthal.hicann.floating_gates = fgc
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
                shift = nc.at(neuron_id).at(21).apply(0) * 1800./1023. * 1e-3 # Convert to mV
                shifts[neuron] = shift
            except:
                self.logger.WARN("No readout shift calibration for neuron {} found. Using unshifted values.".format(neuron))
                shifts[neuron] = 0
        return shifts

    def get_calibrated(self, parameters, nc, bc, coord):
        """ Takes a parameter dictonary, a calibration and a coordinate
            to return the calibrated parameter dictionary.

            Args:
                parameters: All parameters of the experiment
                ncal: which calibration should be used?
                coord: neuron coordinate
                param: parameter that should be calibrated
            Returns:
                Calibrated DAC value (if calibration existed)
        """
        value = parameters[param]
        dac_value = value.toDAC().value #implicit range check!
        dac_value_uncalibrated = dac_value
        if nc and value.apply_calibration:
            try:
                calibration = nc.at(param)
                dac_value = int(round(calibration.apply(dac_value)))
            except (RuntimeError, IndexError),e:
                pass
            except Exception,e:
                raise e

        if dac_value < 0 or dac_value > 1023:
            if self.target_parameter == neuron_parameter.I_gl: # I_gl handled in another way. Maybe do this for other parameters as well.
                msg = "Calibrated value for {} on Neuron {} has value {} out of range. Value clipped to range."
                self.logger.WARN(msg.format(param.name, coord.id(), dac_value))
                if dac_value < 0:
                    dac_value = 10      # I_gl of 0 gives weird results --> set to 10 DAC
                else:
                    dac_value = 1023
            else:
                msg = "Calibrated value for {} on Neuron {} has value {} out of range. Value clipped to range."
                self.logger.WARN(msg.format(param.name, coord.id(), dac_value))
                if dac_value < 0:
                    dac_value = 0      # I_gl of 0 gives weird results --> set to 10 DAC
                else:
                    dac_value = 1023

        return int(round(dac_value))

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


#class BuildVsyntcCalibration(BaseExperimentBuilder):
#    
#    def generate(self):
#
#        meassurements = []
#        # TODO make
#        analyses = [
#                AnalysePSP(threshold, ...),
#                ]
#        return meassurments, analyses
#
