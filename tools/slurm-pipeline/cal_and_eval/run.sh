#!/bin/bash -e

# due to "trap" this is called after the script exits (regardless of exit code)
cleanup() {
    WAFER=$1
    HICANN=$2
    OUTDIR=$3
    tmp_error_log=$4
    report=$5
    resume_number=$6
    exit_code=$7
    cal_eval=$8
    echo
    echo Cleaning up
    echo
    set_status $@
    sleep 10 # because sometimes the shared memory file still exists
    fpga=$(get_fpga_from_hicann $HICANN $WAFER)
    reticle_init.py --wafer $WAFER --fpga $fpga --zero-floating-gate --defects_path "${STHAL_DEFECTS_PATH:-/wang/data/calibration/brainscales/default}"
    exit $exit_code
}

get_fpga_from_hicann() {
    fpga=$(python "-" "$1" "$2" <<EndOfPythonCode
import sys
import pyhalco_hicann_v2 as C
from pyhalco_common import Enum
hicann = int(sys.argv[1])
wafer = int(sys.argv[2])
print (C.HICANNGlobal(C.HICANNOnWafer(Enum(hicann)),
             C.Wafer(Enum(wafer))).toFPGAOnWafer().value())
EndOfPythonCode
)
    echo $fpga
}

set_status() {
    # write the error that occured (or no error) to report.json
    WAFER=$1
    HICANN=$2
    OUTDIR=$3
    tmp_error_log=$4
    report=$5
    resume_number=$6
    exit_code=$7
    cal_eval=$8

    complete_error_log="${OUTDIR}/complete_error_log_w${WAFER}_h${HICANN}.log"
    no_error_message="No errors occured on run ${resume_number}. (OR PROCESS WAS SCANCELED OR RECVFROM(err=XX))"
    error_headline="Error on run ${resume_number} (${cal_eval}):"

    # first search for FPGA and scancel error
    for error_msg in "recvfrom (err=11)" "slurmstepd: error:"; do
        stdout_file=$(find ${OUTDIR} -name "slurm_${cal_eval}*_run${resume_number}.log")
        if [[ -f $stdout_file ]]; then
            error=$(grep "${error_msg}" ${stdout_file}) || error=
            if [[ -n $error ]]; then
                echo "$error" >> $tmp_error_log
            fi
        fi
    done
    python write_report.py $WAFER $HICANN "$tmp_error_log" $resume_number $exit_code "$cal_eval" $report

    printf "\n${error_headline}\n\n" >> $complete_error_log
    if [[ ! -s $tmp_error_log ]]; then
        echo $no_error_message >> $complete_error_log
    else
        cat $tmp_error_log >> $complete_error_log
    fi
    echo >> $complete_error_log
    rm $tmp_error_log
}

# parse input arguments and set variables
. ../parse_input.sh $@

tmp_error_log="${OUTDIR}/tmp_error_log.log"
report="${OUTDIR}/report_w${WAFER}_h${HICANN}.json"

# check for existing logfiles, set name of current cake logfile
log_prefix=submit_w${WAFER}_h${HICANN}_run
resume_number=$(grep -Po 'run\K[^.]+' <<< $(find "${OUTDIR}" -maxdepth 1 -name "${log_prefix}*" | sort | tail -n1))
logfile="cake_${CAL_EVAL}_w${WAFER}_h${HICANN}_run${resume_number}_$(date +"%Y-%m-%d_%H%M%S").log"

# first check if calibration/evaluation has already completed
exit_code=$(python "-" "$report" "$CAL_EVAL" <<EndOfPythonCode
import sys
import os
import json
if os.path.isfile(sys.argv[1]):
    with open(sys.argv[1], 'r') as report:
        cal_eval_report = json.load(report).get(sys.argv[2], None)
    if cal_eval_report is not None:
        if cal_eval_report['complete']:
            print 0
    else:
        print 1
EndOfPythonCode
)
if [[ $NEW == false && $REDO_FITS == false && $exit_code == 0 ]]; then
    echo "${CAL_EVAL} has already completed. Exit without starting the ${CAL_EVAL} run."
    python write_report.py $WAFER $HICANN "$tmp_error_log" $resume_number $exit_code \
                           "$CAL_EVAL" $report
    exit 0
fi

# tells the script to call cleanup on exit
trap "cleanup $WAFER $HICANN $OUTDIR $tmp_error_log $report $resume_number 1 $CAL_EVAL" EXIT

HICANN_VERSION=$(query_hwdb hicann-version --hicann "$HICANN" --wafer "$WAFER")
echo "HICANN version: "  ${HICANN_VERSION}

if [[ "${CAL_EVAL}" == "calib" ]]; then
    cal_eval_long="calibration"
    if [[ "$HICANN_VERSION" -eq "4" ]]; then
            CALIB_CONFIG="v4_params.py"
    else
            CALIB_CONFIG="fastcalibration_parameters.py"
    fi
elif [[ "${CAL_EVAL}" == "eval" ]]; then
    cal_eval_long="evaluation"
    if [[ "$HICANN_VERSION" -eq "4" ]] ; then
            CALIB_CONFIG="v4_eval.py"
    else
            CALIB_CONFIG="evaluation_parameters.py"
    fi
fi

if [[ -n "$WAFER_CFG" ]]; then
    wafer_cfg_overwrite="--overwrite wafer_cfg string ${WAFER_CFG}"
else
    wafer_cfg_overwrite=""
fi

# If BACKEND_PATH is given, calibration results from this path are copied
# to the folder of the current experiment. This usage is intended for
# evaluation of already existing calibrations.
if [[ -n "${BACKEND_PATH}" && "${CAL_EVAL}" == "eval" ]]; then
    if ! [[ -d "${OUTDIR}/backends" ]]; then
        mkdir -v ${OUTDIR}/backends
    fi
    cp -v ${BACKEND_PATH}/*${WAFER}*${HICANN}*xml ${OUTDIR}/backends
fi

aout_overwrite="--overwrite analog AnalogOnHICANN ${AOUT}"

DEVICE_TYPE=$(query_hwdb device-type --wafer "$WAFER")

echo "Device type: " ${DEVICE_TYPE}

if [[ "$DEVICE_TYPE" != "CubeSetup" ]]; then

    experiment_dir=$(find "${OUTDIR}" -name "${cal_eval_long}_f*" | sort | tail -n1)
    if [[ $NEW == true || -z "$experiment_dir" ]]; then
        { #try
          # make a copy of the config file with the resume (run) number in the name
          # to allow to check later what was really used
          cp -v $OUTDIR/$CALIB_CONFIG $OUTDIR/${CALIB_CONFIG}.run${resume_number}

          cake_run_calibration --hicann "$HICANN" --wafer "$WAFER" \
          --outdir "$OUTDIR" --logfile "${OUTDIR}/${logfile}" \
          "${OUTDIR}/${CALIB_CONFIG}" \
            $aout_overwrite 2> >(tee $tmp_error_log >&2) \
          && exit_code=0
        } || { #failed => exit_code=1
            exit_code=1
        }
    else
        { #try
          if [[ $REDO_FITS == true ]] ; then
              skip_fits_flag=""
          else
              skip_fits_flag="--skip_fits"
          fi
          cake_resume ${skip_fits_flag} "${experiment_dir}" \
            --logfile "${OUTDIR}/${logfile}" \
            2> >(tee $tmp_error_log >&2) \
          && exit_code=0
        } || { #failed => exit_code=1
            exit_code=1
        }
    fi
else
    echo "CUBEs not supported ATM"
    exit 1
fi

# tells the script to call cleanup on exit
trap "cleanup $WAFER $HICANN $OUTDIR $tmp_error_log $report $resume_number $exit_code $CAL_EVAL" EXIT
