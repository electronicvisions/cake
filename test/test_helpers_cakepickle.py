#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the cakepickle helper.
"""

import unittest
import shutil
import os
import tempfile

from pycake.helpers.cakepickle import Experimentreader, Experiment


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

    def test_list(self):
        reader = Experimentreader(self.basedir)
        experiments = reader.list_experiments(prnt=False)
        self.assertIsInstance(experiments, list)


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.basedir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.basedir)

    def test_init(self):
        # this will fail if the directory is not an experiment directory...
        # basedir is empty
        experiment = Experiment(self.basedir)
        self.assertEqual(experiment.workdir, self.basedir)


if __name__ == "__main__":
    unittest.main()
