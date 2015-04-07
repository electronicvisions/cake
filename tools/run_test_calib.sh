#!/bin/bash -ev

WAFER=1
HICANN=288
OUTDIR=./build/results_calibration
PLOTDIR=$OUTDIR/plots

mkdir -p $OUTDIR
mkdir -p $PLOTDIR

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

find $OUTDIR

unset -v latest_calibration
for d in $OUTDIR/calibration*; do
  [[ $d -nt $latest_calibration ]] && latest_calibration=$d
done

unset -v latest_evaluation
for d in $OUTDIR/evaluation*; do
  [[ $d -nt $latest_evaluation ]] && latest_evaluation=$d
done

bin/tools/make_plots.py $latest_calibration $latest_evaluation $HICANN $OUTDIR/backends --wafer $WAFER --outdir $PLOTDIR
cp cake/pycake/bin/overview.html $PLOTDIR
