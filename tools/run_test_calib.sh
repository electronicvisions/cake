#!/bin/bash -ev

module load mongo

if [ -z $LD_LIBRARY_PATH ]; then
    export LD_LIBRARY_PATH=$PWD/lib
else
    export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
fi
if [ -z $PYTHONPATH ]; then
    export PYTHONPATH=$PWD/lib
else
    export PYTHONPATH=$PWD/lib:$PYTHONPATH
fi

bin/tools/run_calibration.py cake/config/fastcalibration_parameters.py --wafer 1 --hicann 288 --outdir ~/build/results_calibration
