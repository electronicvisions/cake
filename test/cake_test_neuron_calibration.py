#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This test converts PyNN parameters to DAC using calibtic
using fake calibration data and checks whether DAC values
are calibrated according to the fake calibration.
"""

import unittest

import pycellparameters
import pyNN.standardmodels.cells as pynn
from pycake.helpers.calibtic import Calibtic

import pyhalbe
import pycalibtic
from pyhalco_common import Enum
import pyhalco_hicann_v2 as Coordinate
neuron_parameter = pyhalbe.HICANN.neuron_parameter


class TestNeuronCalibration(unittest.TestCase):
    def setUp(self):
        # Configuration Section
        # These settings match the fake calibration data.
        self.config = {
            "c_wafer": Coordinate.Wafer(Enum(0)),
            "c_hicann": Coordinate.HICANNOnWafer(Enum(280)),
            "c_nrn": Coordinate.NeuronOnHICANN(Enum(23)),
            "hw_vreset": 0.5,  # 500mV
            "xmlbackendpath": "share/cake/fake_calib/"  # use fake data
        }

    def test_apply_calibration(self):
        """Load and apply fake calibration."""

        # load calibration data
        calib = Calibtic(self.config["xmlbackendpath"], self.config["c_wafer"], self.config["c_hicann"])
        neuron_cal = calib.nc
        shared_cal = calib.bc

        # create desired parameters via PyNN
        pynn_parameters = pynn.IF_cond_exp.default_parameters
        pynn_parameters['v_reset'] = -70.
        pynn_parameters['e_rev_I'] = -60.
        pynn_parameters['v_rest'] = -50.
        pynn_parameters['e_rev_E'] = -40.
        parameters = pycellparameters.IF_cond_exp(pynn_parameters)

        # apply calibration
        c_nrn = self.config["c_nrn"]
        ncal_params = pycalibtic.NeuronCalibrationParameters()
        hwparam_n = neuron_cal.applyNeuronCalibration(parameters, c_nrn.toEnum().value(), ncal_params)

        block_id = int(c_nrn.toSharedFGBlockOnHICANN().toEnum())
        hwparam_s = shared_cal.applySharedCalibration(self.config["hw_vreset"], block_id)

        # modify FGControl using calibrated parameters
        fgc = pyhalbe.HICANN.FGControl()

        hwparam_n.toHW(c_nrn, fgc)
        hwparam_s.toHW(c_nrn.toSharedFGBlockOnHICANN(), fgc)

        unmodified_fgc = pyhalbe.HICANN.FGControl()
        self.assertNotEqual(unmodified_fgc, fgc)

        """
        more useful assertions would check values, which
        involves transformations from pyNN to HW,
        including readout/V_reset shift
        """

if __name__ == "__main__":
    import pylogging
    from pysthal.command_line_util import init_logger

    init_logger(pylogging.LogLevel.WARN, [
        (Calibtic.logger.getName(), pylogging.LogLevel.INFO),
    ])
    unittest.main()
