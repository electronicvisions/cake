#!/bin/bash -e

# parse input arguments and set variables
. ../parse_input.sh $@

FIGDIR="figures"

if [[ $CAL_EVAL == calib ]]; then
    calib_folder_pre="calibration_f"
elif [[ $CAL_EVAL == eval ]]; then
    calib_folder_pre="evaluation_f"
    CAL_EVAL="evaluation"
else
    echo "CAL_EVAL has to be 'calib' or 'eval'"
    exit 1
fi

hdf_path=$(find "${OUTDIR}" -name "results.h5" | grep "$calib_folder_pre" | sort | tail -n1)
if [[ -z $hdf_path ]]; then
    echo "Path of the HDF file store could not be found. Aborting."
    exit 1
else
    hdf_path=$(dirname $hdf_path)
fi

BACKENDDIR=$(find "${OUTDIR}" -name "backends" | sort | tail -n1)

cake_make_plots "${OUTDIR}" --$CAL_EVAL "$hdf_path" --outdir $FIGDIR --wafer $WAFER \
                --hicann $HICANN --backend_path $BACKENDDIR

DATE=$(date --iso)
sed -i.bak -e \
"s/<body>/<body><h1>W${WAFER}.H${HICANN} -- ${DATE} HICANN version ${HICANN_VERSION}, ${HICANN_LABEL}<\/h1>/" \
${OUTDIR}/${FIGDIR}/overview.html
