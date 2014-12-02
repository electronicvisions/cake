#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the cakepickle helper.
"""

import unittest
import shutil
import os
import tempfile

from pycake.helpers.cakepickle import Experimentreader
from pyhalbe import Coordinate as C


class TestPickleHelper(unittest.TestCase):
    def setUp(self):
        self.basedir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.basedir)

    def test_init(self):
        # pwd
        reader = Experimentreader()
        self.assertEqual(reader.workdir, os.getcwd())

        # custom workdir
        reader = Experimentreader(self.basedir)
        self.assertEqual(reader.workdir, self.basedir)


if __name__ == "__main__":
    unittest.main()
