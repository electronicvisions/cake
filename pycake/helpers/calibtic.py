import pycalibtic
import pywrapstdvector
import pylogging
import os
import getpass
import copy
import time
import numpy as np

import Coordinate
from pyhalbe.HICANN import neuron_parameter, shared_parameter
from pycake.helpers.units import DAC, Volt, Ampere, Unit, Second

def create_pycalibtic_transformation(coefficients, domain=None, trafo_type=pycalibtic.Polynomial):
    """Create a pycalibtic.Polynomial from a list of coefficients.

    Order: [c0, c1, c2, ...] resulting in c0*x^0 + c1*x^1 + c2*x^2 + ...

    Domain should be a tuple wit min and max possible hardware value"""
    # Make standard python list to have the right order
    if coefficients is None:
        return None
    coefficients = list(coefficients)
    data = pywrapstdvector.Vector_Double(coefficients)
    if domain is None:
        return trafo_type(data)
    return trafo_type(data, domain[0], domain[1])

class Calibtic(object):
    logger = pylogging.get("pycake.calibtic")

    def __setstate__(self, dic):
        # For old pickles: just set bigcap and speedup to default values
        # Otherwise, unpickling of old calibtic helpers does not work
        if 'pll' not in dic:
            dic['pll'] = 0.0
        if 'ideal_nc' not in dic:
            dic['ideal_nc'] = pycalibtic.NeuronCollection()
            dic['ideal_nc'].setDefaults()
        if 'ideal_bc' not in dic:
            dic['ideal_bc'] = pycalibtic.BlockCollection()
            dic['ideal_bc'].setDefaults()
        if not (dic.has_key('bigcap') and dic.has_key('speedup')):
            dic['bigcap'] = True
            dic['speedup'] = 'normal'
            dic['hc'] = None
        if 'hc' not in dic:
            self.hc = self.load_calibration()

    def __init__(self, path, wafer, hicann, pll=100e6, backend_type='xml'):
        self.path = self.make_path(path)
        self.wafer = wafer
        self.hicann = hicann
        self.pll = pll

        self.ideal_nc = pycalibtic.NeuronCollection()
        self.ideal_nc.setDefaults()
        self.ideal_bc = pycalibtic.BlockCollection()
        self.ideal_bc.setDefaults()
        self.hc = self.load_calibration(backend_type)

    def get_backend(self, type='xml'):
        if type == 'xml':
            lib = pycalibtic.loadLibrary('libcalibtic_xml.so')
            backend = pycalibtic.loadBackend(lib)
            backend.config('path', self.path)
            backend.init()
            return backend
        elif type == 'binary':
            lib = pycalibtic.loadLibrary('libcalibtic_binary.so')
            backend = pycalibtic.loadBackend(lib)
            backend.config('path', self.path)
            backend.init()
            return backend
        else:
            raise ValueError("unknown backend type")

    @staticmethod
    def make_path(path):
        path = os.path.expanduser(path)
        if not os.path.isdir(path):
            Calibtic.logger.INFO("Creating backend path {}".format(path))
            os.makedirs(path)
        return path

    def clear_calibration(self):
        """ Clears all calibration data.
        """
        name = self.get_calibtic_name()
        fullname = name + ".xml"
        fullpath = os.path.join(self.path, fullname)
        if os.path.isfile(fullpath):
            msg = "Clearing calibration data by removing file {}"
            self.logger.INFO(msg.format(fullpath))
            os.rename(fullpath, fullpath + time.strftime('.%y%m%d_%H%M%S.bak'))

    def clear_one_calibration(self, parameter):
        """ Only clears the calibration for one parameter.
        """
        if isinstance(parameter, shared_parameter):
            collection = self.bc
            ids = range(4)
        else:
            collection = self.nc
            ids = range(512)

        for i in ids:
            self.logger.TRACE("Resetting calibration for parameter {} for id {}".format(parameter.name, i))
            collection.at(i).reset(parameter, None)

    def get_calibtic_name(self):
        wafer_id = self.wafer.value()
        hicann_id = self.hicann.id().value()
        name = "w{}-h{}".format(int(wafer_id), int(hicann_id))
        return name

    def load_calibration(self, backend_type='xml'):
        """ Load existing calibration data from backend.
        """

        hc = pycalibtic.HICANNCollection()
        md = pycalibtic.MetaData()

        # load existing calibration
        try:
            name = self.get_calibtic_name()
            backend = self.get_backend(backend_type)
            backend.load(name, md, hc)
            calibration_existed = True
        except RuntimeError, e:
            if "data set not found" not in e.message:
                raise

            # Delete all standard entries.
            #TODO: fix calibtic to use proper standard entries

            nc = hc.atNeuronCollection()
            for nid in range(512):
                nc.erase(nid)
            bc = hc.atBlockCollection()
            for bid in range(4):
                bc.erase(bid)
            hc.setPLLFrequency(int(self.pll))

        if hc.getPLLFrequency() != self.pll:
            msg  = "PLL stored in HICANNCollection {} MHz != {} MHz set here"
            self.logger.ERROR(
                msg.format(hc.getPLLFrequency()/1e6, self.pll/1e6))
        return hc

    @property
    def nc(self):
        """Returns a pycalibtic.NeuronCollection"""
        return self.hc.atNeuronCollection()

    @property
    def bc(self):
        """Returns a pycalibtic.BlockCollection"""
        return self.hc.atBlockCollection()

    def get_calib_id(self, param_name,
                     target_bigcap,
                     target_I_gl_speedup,
                     target_I_gladapt_speedup,
                     target_I_radapt_speedup):
        """ try to access the calibration id by the same name,
            if it fails, try with speedup and bigcap suffix

            Args:
                 target_bigcap: True/False
                 target_*_speedup: "slow"/"fast"

            works only for neuron_parameter
        """

        calibs = pycalibtic.NeuronCalibrationParameters.Calibrations

        try:
            calib_id = getattr(calibs, param_name)
        except AttributeError:
            slow = locals()["target_{}_speedup".format(param_name)] == "slow"
            fast = locals()["target_{}_speedup".format(param_name)] == "fast"
            suffix = "slow{}_fast{}_bigcap{}".format(int(slow), int(fast), int(target_bigcap))
            calib_id = getattr(calibs, "{}_{}".format(param_name, suffix))
        return calib_id

    def write_calibration(self, parameter, trafos,
                          target_bigcap,
                          target_I_gl_speedup,
                          target_I_gladapt_speedup,
                          target_I_radapt_speedup,
                          backend_type='xml'):
        """ Writes calibration data to backend

            Args:
                parameter: which neuron or hicann parameter
                trafos: dict { coord : pycalibtic.transformation }
        """
        name = self.get_calibtic_name()

        for coord, trafo in trafos.iteritems():

            if trafo is None:
                continue

            if isinstance(parameter, shared_parameter) and isinstance(coord, Coordinate.FGBlockOnHICANN):
                collection = self.bc
                cal = pycalibtic.SharedCalibration()
                calib_id = parameter
                param_name = parameter.name
            elif isinstance(parameter, neuron_parameter) and isinstance(coord, Coordinate.NeuronOnHICANN):
                collection = self.nc
                cal = pycalibtic.NeuronCalibration()
                param_name = parameter.name
                calib_id = self.get_calib_id(param_name, target_bigcap, target_I_gl_speedup, target_I_gladapt_speedup, target_I_radapt_speedup)
            elif parameter is 'readout_shift':
                collection = self.nc
                cal = pycalibtic.NeuronCalibration()
                calib_id = pycalibtic.NeuronCalibrationParameters.Calibrations.ReadoutShift
                param_name = parameter
            else:
                raise RuntimeError("Cannot write calibration for parameter {} of coordinate {}".format(parameter, coord))

            index = coord.id().value()

            if not collection.exists(index):
                collection.insert(index, cal)
            collection.at(index).reset(calib_id, trafo)
            self.logger.TRACE("Resetting coordinate {} parameter {} to {}".format(coord, param_name, trafo))
        md = pycalibtic.MetaData()
        backend = self.get_backend(backend_type)
        backend.store(name, md, self.hc)

    def get_calibration(self, coord, use_ideal=False):
        """ Returns NeuronCalibration or SharedCalibration object for one coordinate.

            If a collection is not found und use_ideal is set to True, returns an ideal calibration
            This is turned off by default.
            If no calibration is found for a coordinate, it returns an empty calibration.
        """
        c_id = coord.id().value()

        if isinstance(coord, Coordinate.FGBlockOnHICANN):
            collection = self.bc
            calibration = pycalibtic.SharedCalibration
        else:
            collection = self.nc
            calibration = pycalibtic.NeuronCalibration

        if collection.exists(c_id) and not use_ideal:
            self.logger.TRACE("Found Calibration for {}".format(coord))
            return collection.at(c_id)
        elif use_ideal:
            self.logger.WARN("No calibration dataset found for {}. Returning ideal calibration.".format(coord))
            if isinstance(coord, Coordinate.FGBlockOnHICANN):
                return self.ideal_bc.at(c_id)
            else:
                return self.ideal_nc.at(c_id)
        else:
            # return empty calibration if none exists
            self.logger.WARN("No calibration dataset found for {}. Returning empty calibration".format(coord))
            return calibration()

    def get_readout_shift(self, neuron):
        """
        """
        calib = self.get_calibration(neuron)
        if not calib:
            return 0.0

        if calib.exists(pycalibtic.NeuronCalibration.Calibrations.ReadoutShift):
            shift = calib.at(pycalibtic.NeuronCalibration.Calibrations.ReadoutShift).apply(0.0)
            self.logger.TRACE("Readout shift for neuron {0}:{1:.2f}".format(neuron, shift))
        else:
            shift = 0.0
            self.logger.WARN("No readout shift found for neuron {}.".format(neuron))

        return shift

    def apply_calibration(self, value, parameter, coord,
                          bigcap,
                          speedup_I_gl,
                          speedup_I_gladapt,
                          speedup_I_radapt,
                          use_ideal=False, report=False,
                          outside_domain_behaviour="CLIP"):
        """ Apply calibration to one value. If no calibration is found, use ideal calibration.
            Also, if parameter value is given in a unit that does not support calibration, e.g.
            I_pl given in Ampere instead of microseconds, apply direct translation to DAC.

            Args:
                value: Value in Hardware specific units (e.g. volt, ampere, seconds)
                parameter: neuron_parameter or shared_parameter type
                coord: neuron coordinate
                outside_domain_behaviour: "CLIP"/"IGNORE"/"THROW"
                                          (see calibtic/trafo/Transformation.h)
            Returns:
                Calibrated DAC value
                If the calibrated value exceeds the boundaries, it is clipped.

                If report_status is set, also a integer is retured, discribing the
                actuale transformation used: 0 ok, 1 ideal, 2 none, and a bool
                indicating if the value is clipped
        """
        if not isinstance(value, Unit):
            value = Unit(value)

        calib = self.get_calibration(coord, use_ideal)
        status = 0
        clipped = False

        if parameter in [neuron_parameter.I_pl,
                         neuron_parameter.I_gl,
                         neuron_parameter.V_syntcx,
                         neuron_parameter.V_syntci]:
            if not isinstance(value, Second):
                raise TypeError("Calibration for {} only valid for seconds.".format(parameter))

        calib_id = parameter if isinstance(parameter,
                                           shared_parameter) else self.get_calib_id(parameter.name,
                                                                                    bigcap,
                                                                                    speedup_I_gl,
                                                                                    speedup_I_gladapt,
                                                                                    speedup_I_radapt)

        if calib.exists(calib_id):
            # Apply calibration
            calib_dac_value = calib.to_dac(value.value, calib_id,
                                           getattr(pycalibtic.Transformation,
                                                   outside_domain_behaviour))
            self.logger.TRACE("Calibrated {} parameter {}: {} --> {} DAC".format(coord, parameter.name, value, calib_dac_value))
        else:
            ideal = self.get_calibration(coord, True)
            calib_dac_value = ideal.to_dac(value.value, calib_id,
                                           getattr(pycalibtic.Transformation,
                                                   outside_domain_behaviour))
            self.logger.WARN("Calibration for {} parameter {} not found. Using ideal transformation -> DAC value {}".format(coord, parameter.name, calib_dac_value))
            status = 2

        calib_dac_value = np.clip(calib_dac_value + value.dac_offset, 0, 1023)

        if report:
            return calib_dac_value, status, clipped
        else:
            return calib_dac_value

    def set_neuron_parameters(self, parameters, neuron, floating_gates, bigcap, speedup_I_gl, speedup_I_gladapt, speedup_I_radapt):
        """Writes floating gate parameters for a single neuron into a sthal
        FloatingGates container. This includes calibration and transformation
        from V or nA to DAC values.

        Parameters:
            parameters: dictionary containg the parameters to be written,
                neuron specific parameters are consumed
            neuron: neuron to be updated
            floating_gates: pysthal.FloatingGates to be updated

        Returns:
            pysthal.FloatingGates with given parameters
        """
        # We take only a shallow copy, because this is faster. The values
        # are copied in the loop below
        neuron_params = parameters.copy()
        # Use pop to consume neuron specific parameters
        neuron_params.update(parameters.pop(neuron, {}))
        for param, value in neuron_params.iteritems():
            if (not isinstance(param, neuron_parameter)) or param.name[0] == '_':
                # e.g. __last_neuron
                continue

            value = copy.deepcopy(value)  # Copy of value, see comment above
            if value.apply_calibration:
                self.logger.TRACE("Applying calibration to coord {} value {}".format(neuron, value))
                value_dac, status, clipped  = self.apply_calibration(
                    value, param, neuron,
                    bigcap, speedup_I_gl, speedup_I_gladapt, speedup_I_radapt,
                    report=True)
            else:
                # special case for unit second and no calibration applied
                if isinstance(value, Second):
                    value_dac, status, clipped  = self.apply_calibration(value, param, neuron,
                                                                         bigcap, speedup_I_gl, speedup_I_gladapt, speedup_I_radapt,
                                                                         use_ideal=True, report=True)
                else:
                    value_dac = value.toDAC().value
            self.logger.TRACE("Setting FGValue of {} parameter {} to {}.".format(neuron, param, value_dac))
            floating_gates.setNeuron(neuron, param, value_dac)

    def set_shared_parameters(self, parameters, block, floating_gates, bigcap, speedup_I_gl, speedup_I_gladapt, speedup_I_radapt):
        """Writes shared floating gate parameters for a single floating gate
        block into a sthal FloatingGates container. This includes calibration
        and transformation from V or nA to DAC values.

        Parameters:
            parameters: dictionary containg the parameters to be written
            block: shared floating gate block to be updated
            floating_gates: pysthal.FloatingGates to be updated

        Returns:
            pysthal.FloatingGates with given parameters
        """
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

            if value.apply_calibration:
                self.logger.TRACE("Applying calibration to coord {} value {}".format(block, value))
                value_dac = self.apply_calibration(value, param, block, bigcap, speedup_I_gl, speedup_I_gladapt, speedup_I_radapt)
            else:
                if type(value) in [Ampere, Volt, DAC]:
                    value_dac = value.toDAC().value
                else:
                    value_dac = self.apply_calibration(value.value, param, block, bigcap, speedup_I_gl, speedup_I_gladapt, speedup_I_radapt, use_ideal=True)

            self.logger.TRACE("Setting FGValue of {} parameter {} to {}.".format(block, param, value))
            floating_gates.setShared(block, param, value_dac)


    def set_calibrated_parameters(self, parameters, neurons, blocks, floating_gates,
                                  bigcap,
                                  speedup_I_gl,
                                  speedup_I_gladapt,
                                  speedup_I_radapt):
        """Writes floating gate parameters into a sthal FloatingGates container.
            This includes calibration and transformation from V or nA to DAC values.

        Parameters:
            parameters: dictionary containg the parameters to be written
            floating_gates: pysthal.FloatingGates to be updated

        Returns:
            pysthal.FloatingGates with given parameters
        """

        for neuron in neurons:
            self.set_neuron_parameters(parameters, neuron, floating_gates, bigcap, speedup_I_gl, speedup_I_gladapt, speedup_I_radapt)
        for block in blocks:
            self.set_shared_parameters(parameters, block, floating_gates, bigcap, speedup_I_gl, speedup_I_gladapt, speedup_I_radapt)
