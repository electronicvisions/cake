#!/bin/bash -e

task=calib-$1

# parse input arguments and set variables
. ../parse_input.sh $SP_ORIGINAL_ARGS -t="calib"

cd ../cal_and_eval/
source submit.sh
