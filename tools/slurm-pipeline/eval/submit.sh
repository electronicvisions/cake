#!/bin/bash -e

task=eval-$1

# parse input arguments and set variables
. ../parse_input.sh $SP_ORIGINAL_ARGS -t="eval"

cd ../cal_and_eval/
source submit.sh
