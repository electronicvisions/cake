#!/bin/bash -ev

if [ $# -ne 3 ]; then
	echo "./run_test_calib.sh WAFER HICANN HICANN_VERSION (2 or 4)"
	exit 1
fi

WAFER=$1
HICANN=$2
HICANN_VERSION=$3
OUTDIR=./build/results_calibration
PLOTDIR=$OUTDIR/plots

echo "outdir: $OUTDIR"
echo "plotdir: $PLOTDIR"

mkdir -p $OUTDIR
mkdir -p $PLOTDIR

# c.f. cake/project.prj
module load mongo
module load yaml-cpp/0.5.2
module load anaconda
module load pynn/0.7.5
module load localdir

find .

cp -v bin/tools/*_parameters.py .
cp -v bin/tools/v4_params.py .
cp -v bin/tools/v4_eval.py .

# issue 1621
export PYTHONPATH=.:$PYTHONPATH

case $HICANN_VERSION in
    2)
		calib_params=fastcalibration_parameters.py
		eval_params=evaluation_parameters.py
        ;;
    4)
		calib_params=v4_params.py
		eval_params=v4_eval.py
        ;;
    *)
        echo "Unknown hicann version: $HICANN_VERSION"
		exit 1
esac

echo "calib_parms: $calib_params"
echo "eval_parms: $eval_params"

# run calibration
bin/tools/run_calibration.py $calib_params --wafer $WAFER --hicann $HICANN --outdir $OUTDIR --overwrite clear bool True --overwrite clear_defects bool True

find $OUTDIR

unset -v latest_calibration
for d in $OUTDIR/calibration*; do
  [[ $d -nt $latest_calibration ]] && latest_calibration=$d
done

# run evaluation
bin/tools/run_calibration.py $eval_params --wafer $WAFER --hicann $HICANN --outdir $OUTDIR

find $OUTDIR

unset -v latest_evaluation
for d in $OUTDIR/evaluation*; do
  [[ $d -nt $latest_evaluation ]] && latest_evaluation=$d
done

# make plots
bin/tools/make_plots.py $latest_calibration $latest_evaluation $HICANN $OUTDIR/backends --wafer $WAFER --outdir $PLOTDIR
cp -v bin/tools/overview.html $PLOTDIR

find $OUTDIR
