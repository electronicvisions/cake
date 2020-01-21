#!/usr/bin/env python
"""Test pycake.config module functionality."""

import unittest
import os
from pycake.config import Config
from pyhalco_common import iter_all
import pyhalco_hicann_v2 as Coordinate

class TestConfig(unittest.TestCase):
    """Test Config class."""

    def test_empty_config(self):
        """
        Initialize with empty dictionary.
        """
        cfg = Config(None, {})
        self.assertIsInstance(cfg, Config)
        cfg2 = cfg.copy("other name")
        self.assertEqual(cfg2.config_name, "other name")
        self.assertEqual(cfg.get_PLL(), 100e6)

    def test_example_A(self):
        config_dir = os.path.dirname(os.path.realpath(__file__))
        cfg = Config("random-name",
                     os.path.join(config_dir, "example_config_A.py"))

        # preset values
        self.assertEqual(cfg.get_target(), "random-name")
        self.assertEqual(cfg.get_blocks(), [block for block in iter_all(Coordinate.FGBlockOnHICANN)])
        self.assertEqual(cfg.get_config("foo"), "foo-value")
        self.assertEqual(cfg.get_config_with_default("foo", "other-value"), "foo-value")
        self.assertEqual(cfg.get_config_with_default("missing", "default-value"), "default-value")
        self.assertEqual(cfg.get_neurons(), [neuron for neuron in iter_all(Coordinate.NeuronOnHICANN)])
        self.assertEqual(cfg.get_steps(), "steps-value")
        self.assertEqual(cfg.get_folder(), "folder-value")
        self.assertEqual(cfg.get_calibtic_backend(), ("backend-value", "w17-h222"))

        # set-get values
        cfg.set_config("foo-set", "set-value")
        self.assertEqual(cfg.get_config("foo-set"), "set-value")
        cfg.set_config("foo-set", "set-value2")
        self.assertEqual(cfg.get_config("foo-set"), "set-value2")

        # default values
        self.assertEqual(cfg.get_wafer_cfg(), "")
        self.assertEqual(cfg.get_enabled_calibrations(), [])
        self.assertEqual(cfg.get_sim_denmem(), None)
        self.assertEqual(cfg.get_PLL(), 100e6)

    def test_example_B(self):
        config_dir = os.path.dirname(os.path.realpath(__file__))
        cfg = Config("random-name",
                     os.path.join(config_dir, "example_config_B.py"))

        self.assertEqual(cfg.get_wafer_cfg(), "foo.xml")
        self.assertEqual(cfg.get_enabled_calibrations(), ["foo", "bar", "baz"])
        self.assertEqual(cfg.get_sim_denmem(), "vtitan:8123")
        self.assertEqual(cfg.get_PLL(), 125e6)

if __name__ == '__main__':
    unittest.main()
