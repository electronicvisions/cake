#!/bin/bash -ev

# parse input arguments and set variables
. ../parse_input.sh $@

env

for param_file in base_parameters.py  evaluation_parameters.py fastcalibration_parameters.py \
        v4_eval.py v4_params.py; do
    if  [[ ! -f "${NMPM_CALIB_CONFIG}/$param_file" ]]; then
        echo "NMPM_CALIB_CONFIG must contain $param_file"
        exit 1
    fi
done

# copy parameter files to destination
# TODO: Smarter copying, only the files needed instead of all files
cp -v ${NMPM_CALIB_CONFIG}/* ${OUTDIR} 2>/dev/null || :

DEVICE_TYPE=$(query_hwdb device-type --wafer "${WAFER}")
if [[ "${DEVICE_TYPE}" == "CubeSetup" ]] ; then
    echo "CUBEs not supported at the moment"
    exit 1
fi

# Call to redman_wafer_has_hicann.py exits with 1 if HICANN is not available,
#  leading to pipeline task termination.
echo "Checking if HICANN is available..."
redman_wafer_has_hicann.py --wafer "${WAFER}" --hicann "${HICANN}" --defects_path "${STHAL_DEFECTS_PATH:-/wang/data/calibration/brainscales/default}"

# Call to redman_hicann_has_highspeed.py exits with 1 if HICANN does not have
#  high speed comm available, leading to pipeline task termination.
echo "Checking if HICANN has HighSpeed comm..."
redman_hicann_has_highspeed.py --wafer "${WAFER}" --hicann "${HICANN}" --defects_path "${STHAL_DEFECTS_PATH:-/wang/data/calibration/brainscales/default}"

echo This is a version $(query_hwdb hicann-version --hicann "${HICANN}" --wafer "${WAFER}") HICANN
