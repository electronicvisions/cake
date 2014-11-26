#!/bin/bash

############################################
# Runs coverage.py for code coverage report
############################################

COV=python-coverage


# delete old data
$COV erase

# run code
$COV run 'test/run_all_tests.py'

# report result
$COV report

# generate HTML report
$COV html

## generate XML report
#$COV xml
