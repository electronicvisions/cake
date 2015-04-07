#!/bin/bash -ev

WAFER=1
HICANN=288
OUTDIR=./build/results_calibration

# c.f. cake/project.prj
module load numpy/1.8.0
module load pyfftw
module load pynn
module load localdir

find .

cp -v bin/tools/*_parameters.py .

# issue 1621
export PYTHONPATH=.:$PYTHONPATH

bin/tools/run_calibration.py fastcalibration_parameters.py --wafer $WAFER --hicann $HICANN --outdir $OUTDIR
bin/tools/run_calibration.py evaluation_parameters.py --wafer $WAFER --hicann $HICANN --outdir $OUTDIR

find

find ./build/results_calibration