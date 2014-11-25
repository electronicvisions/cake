#!/usr/bin/env python

"""Run all tests in this file's directory."""

import os
import unittest


loader = unittest.TestLoader()
suite = loader.discover(os.path.dirname(__file__), 'test_*.py')
unittest.TextTestRunner().run(suite)
