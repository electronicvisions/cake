#!/bin/bash -e

## Script to start the calibration pipeline for the HICANNs given in $2.
## Arguments:
## * `-w|--wafer` (int): the wafer enum
## * `--hicanns` (int): HICANN(s) to calibrate ( -1 starts for all HICANNs, mutliple
##                    HICANNs can be given like 1,65,2-5,3,90,100-149
## * `-p|--param-folder` (str): directory where all parameter files (v4_params.py etc.) are in
## * `-o|--outdir` (str): directory to save the calibrations to. A new folder is created
##                      in this directory for each HICANN
## * `--firstStep` (str, optional): start pipeline at this step
## * `--lastStep` (str, optional): end pipeline at this step
## * `--new` (optional): run the pipeline even if there are already completed calibrations in outdir
## * `--redo_fits` (optional): ask cake_resume to redo fits. Default is to skip already performed fits.
## ##### Example:
## ```
## sbatch cake_calib_hicanns -w 21 --hicanns -1 -p /path/to/v4_params_etc/ \
## -o `readlink -f 20170510_full_wafer`
## ```
## Remember to load slurm-pipeline (README)

#SBATCH --time=00:59:00 # Time limit hrs:min:sec
#SBATCH --output=slurm_pipeline_%j.out # Standard output and error log
#SBATCH --partition=short

trap "exit 5" SIGINT SIGTERM

# first check if the specification file is in the current directory
if ! [[ -f $(pwd)/specification.json ]]; then
    echo "No specification.json found. Default specification is in cake/tools/slurm-pipeline. If
          you do not intend to use a special specification, switch to this folder."
    exit 7
fi

# parse arguments
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -w|--wafer)
    WAFER="$2"
    shift # past argument
    ;;
    --hicanns)
    HICANNS="$2"
    shift # past argument
    ;;
    -p|--param-folder)
    NMPM_CALIB_CONFIG="$2"
    shift # past argument
    ;;
    -o|--outdir)
    OUTDIR="$2"
    shift # past argument
    ;;
    -b|--backend)
    BACKEND="$2"
    shift # past argument
    ;;
    --fpgas_to_exclude)
    FPGAS="$2"
    shift # past argument
    ;;
    --firstStep)
    FIRST_STEP="--firstStep $2"
    shift # past argument
    ;;
    --lastStep)
    LAST_STEP="--lastStep $2"
    shift # past argument
    ;;
    --switch_aouts)
    SWITCH_AOUTS=true
    ;;
    --new)
    NEW=true
    ;;
    --redo_fits)
    REDO_FITS=true
    ;;
    *)
    ;;
esac
shift # past argument or value
done
if ! [[ $WAFER -gt -1 && ! -z $WAFER ]] ; then
    echo "WAFER is wrong (-w x, x in [0,inf))"
    exit 3
else
    echo "WAFER is $WAFER"
fi
if [[ -z $HICANNS ]] ; then
    echo "HICANNS variable should not be empty."
    exit 3
else
    echo "HICANNS is $HICANNS"
fi
if [[ -z $NMPM_CALIB_CONFIG ]] ; then
    echo "NMPM_CALIB_CONFIG variable should not be empty."
    exit 3
else
    echo "NMPM_CALIB_CONFIG is $NMPM_CALIB_CONFIG"
fi
if [[ -z $OUTDIR ]] ; then
    echo "OUTDIR variable should not be empty."
    exit 3
else
    echo "OUTDIR is $OUTDIR"
fi
if ! [[ -z $FPGAS ]]; then
    echo "FPGAS is $FPGAS"
fi
if ! [[ $SWITCH_AOUTS = true ]] ; then
    SWITCH_AOUTS=false
fi
if ! [[ $NEW = true ]] ; then
    NEW=false
fi
if ! [[ $REDO_FITS = true ]] ; then
    REDO_FITS=false
fi
echo "SWITCH_AOUTS is $SWITCH_AOUTS"
echo "NEW is $NEW"
echo "REDO_FITS is $REDO_FITS"
if ! [[ -z ${BACKEND} ]]; then
    echo "BACKEND is $BACKEND"
    if [[ $NEW = true || $REDO_FITS = true ]]; then
        echo "Backend path -b/--backend should not be provided with --new or --redo_fits flags."
        exit 3
    fi
    backend_flags="--skip calib --skip plot_calib"
else
    backend_flags=""
fi

#global arrays to determine hicanns to start the calib for
parsed_hicann_array=()
excluded_hicanns_array=()

start_calib() {
    hicann=$1
    switch_aouts=$2
    # TODO: more sophisticated calculation of aout based on already used aouts, available aouts
    if [[ $switch_aouts = true ]]; then
        aout=$((-1*(${hicann} % 2)+1))
    else
        aout=$(($hicann % 2))
    fi

    # with leading and trailing spaces for scriptArgs argument, cf. https://bugs.python.org/issue9334
    slurm-pipeline.py --specification specification.json \
        $FIRST_STEP $LAST_STEP --output \
        ${OUTDIR}/w${WAFER}_h${hicann}/w${WAFER}_h${hicann}_aout${aout}_$(date \
         +"%Y-%m-%d_%H%M%S").json \
        ${backend_flags} \
        --scriptArgs w=$WAFER hicann=$hicann a=$aout p=$NMPM_CALIB_CONFIG n=$NEW r=$REDO_FITS \
        b=${BACKEND} \
        outdir=`readlink -f ${OUTDIR}/w${WAFER}_h${hicann}` \
     && echo "Starting calibration job on wafer $WAFER for HICANN $hicann on aout $aout"
}

# sets the variable parsed_hicann_array
parse_hicanns() {
    hicanns=$1
    if [[ $hicanns == -1 ]];then
        parsed_hicann_array=($(seq 0 383))
    else
        hicann_array=$(echo $hicanns | tr "," " ")
        for hc in $hicann_array; do
            if [[ $hc =~ "-" ]] ; then
                hc_range=($(echo $hc | tr "-" " "))
                hc_expanded=$(seq ${hc_range[0]} ${hc_range[1]})
                parsed_hicann_array+=($hc_expanded)
            else
                parsed_hicann_array+=($hc)
            fi
        done
    fi
}

# sets the variable excluded_hicanns_array
parse_fpgas() {
    if ! [[ "$1" = '' ]]; then
        excluded_hicanns_array=($(python "-" "$1" <<EndOfPythonCode
import sys
import Coordinate as C
fpgas_str = sys.argv[1]
fpgas = []
for f in fpgas_str.split(','):
    f_split = f.split('-')
    if len(f_split) == 1:
        fpgas.append(int(f_split[0]))
    else:
        fpgas += range(int(f_split[0]), int(f_split[1])+1)
hicanns = []
for fpga in fpgas:
    wafer = 33
    hicanns += [hc_glob.toHICANNOnWafer().toEnum().value()
                for hc_glob in C.FPGAGlobal(C.FPGAOnWafer(C.Enum(fpga)),
                                                 C.Wafer(wafer)).toHICANNGlobal()]
print " ".join([str(hicann) for hicann in hicanns])
EndOfPythonCode
))
    fi
}

# we collect HICANN enums if the calibration could not be started for the error message at the end
parse_hicanns $HICANNS
parse_fpgas $FPGAS

# difference between given hicanns and fpgas
hicanns_to_start=$(echo ${parsed_hicann_array[@]} ${excluded_hicanns_array[@]} \
                    ${excluded_hicanns_array[@]} | tr ' ' '\n' | sort | uniq -u)

#TODO: Shuffle is used here to start the jobs faster for the different AOUTs
# it would be preferable to circle around the wafer or something like that
hicanns_to_start=($(shuf -e ${hicanns_to_start[@]}))

failed_hicanns=()
for hicann in ${hicanns_to_start[@]}; do
    # make directory, check if creation worked
    if ! mkdir -p "${OUTDIR}/w${WAFER}_h${hicann}"; then
        echo Cannot create "${OUTDIR}/w${WAFER}_h${hicann}". Aborting.
    else
        # start the pipeline
        start_calib $hicann $SWITCH_AOUTS || failed_hicanns+=($hicann)
    fi
done

if [[ -n $failed_hicanns ]]; then
    echo "Could not start calibration pipeline for the following HICANNs:
          ${failed_hicanns[*]}. Maybe there are too many jobs in the queue?"
fi
