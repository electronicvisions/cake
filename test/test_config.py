#!/usr/bin/env python
"""Test pycake.config module functionality."""

import unittest
from pycake.config import Config


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


if __name__ == '__main__':
    unittest.main()
