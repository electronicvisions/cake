#!/bin/bash

############################################
# Runs coverage.py for code coverage report
############################################

# delete old data
coverage erase

# run code
coverage run 'test/run_all_tests.py'

# report result
coverage report

# generate HTML report
coverage html

## generate XML report
#coverage xml
