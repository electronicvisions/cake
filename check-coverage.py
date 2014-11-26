#!/usr/bin/env python

"""Run all tests and record line coverage."""

import os
import unittest
import coverage


cov = coverage.coverage(
    branch=False,
    cover_pylib=False,
    source=["pycake/"],
)

# erase old data
cov.erase()

# exclude counting some lines
cov.exclude("^import")
cov.exclude("^from")

# start recording
cov.start()

# run all tests
loader = unittest.TestLoader()
cakedir = os.path.dirname(__file__)
testdir = os.path.join(cakedir, "test")
suite = loader.discover(testdir, 'test_*.py')
unittest.TextTestRunner().run(suite)

# stop recording, save results
cov.stop()
cov.save()

## command line report
#cov.report()

# generate html report
cov.html_report(
    include=["*/pycake/*"],
    omit=["*/pycake/bin/*"],
    directory="htmlcov",
    #title="Coverage report",
)
