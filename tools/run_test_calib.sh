#!/bin/bash -ev

WAFER=1
HICANN=288
OUTDIR=./build/results_calibration
PLOTDIR=$OUTDIR/plots

mkdir -p $OUTDIR

# c.f. cake/project.prj
module load numpy/1.8.0
module load pyfftw
module load pynn
module load localdir

find .

cp -v bin/tools/*_parameters.py .

# issue 1621
export PYTHONPATH=.:$PYTHONPATH

# run calibration
#bin/tools/run_calibration.py fastcalibration_parameters.py --wafer $WAFER --hicann $HICANN --outdir $OUTDIR --overwrite clear bool True --overwrite clear_defects bool True

find $OUTDIR

unset -v latest_calibration
for d in $OUTDIR/calibration*; do
  [[ $d -nt $latest_calibration ]] && latest_calibration=$d
done

# run evaluation
#bin/tools/run_calibration.py evaluation_parameters.py --wafer $WAFER --hicann $HICANN --outdir $OUTDIR

find $OUTDIR

unset -v latest_evaluation
for d in $OUTDIR/evaluation*; do
  [[ $d -nt $latest_evaluation ]] && latest_evaluation=$d
done

# make plots
mkdir -p $PLOTDIR
bin/tools/make_plots.py $latest_calibration $latest_evaluation $HICANN $OUTDIR/backends --wafer $WAFER --outdir $PLOTDIR
cp bin/tools/overview.html $PLOTDIR
