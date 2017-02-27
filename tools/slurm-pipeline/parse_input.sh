#!/bin/bash -e
# from https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
# - eval is used to strip single parentheses from SP_ORIGINAL_ARGS (slurm-pipeline variable)
# - slurm-pipeline currently only accepts --scriptArgs parameters in the form "parameter=<value>"
#    (i.e. without hyphen, e.g. "-w=20" is not accepted)

for arg in $SP_ORIGINAL_ARGS
do
eval arg=$arg
case $arg in
    w=*|wafer=*)
    WAFER="${arg#*=}"
    ;;
    hicann=*)
    HICANN="${arg#*=}"
    ;;
    a=*|aout=*)
    AOUT="${arg#*=}"
    ;;
    o=*|outdir=*)
    OUTDIR="${arg#*=}"
    ;;
    b=*|backend=*)
    BACKEND_PATH="${arg#*=}"
    ;;
    p=*|param-folder=*)
    NMPM_CALIB_CONFIG="${arg#*=}"
    ;;
    n=*|n=*)
    NEW="${arg#*=}"
    ;;
    r=*|r=*)
    REDO_FITS="${arg#*=}"
    ;;
esac
done

for arg in $@
do
case $arg in
    -t=*|--run-type=*)
    CAL_EVAL="${arg#*=}"
    ;;
esac
done

if ! [[ ${WAFER} -gt -1 && -n ${WAFER} ]] ; then
    echo "WAFER is wrong (-w x, x in [0,inf))"
    exit 3
fi
if ! [[ ${HICANN} -gt -1  &&  ${HICANN} -lt 384 && -n ${HICANN} ]] ; then
    echo "HICANN is wrong (--hicann x, x in [0,383])"
    exit 3
fi
if ! [[ ${AOUT} -gt -1 && ${AOUT} -lt 2 && -n ${AOUT} ]] ; then
    echo "AOUT is wrong (-a x, x in {0,1})"
    exit 3
fi
if [ -z ${OUTDIR} ]; then
    echo "OUTDIR is needed (--outdir "path/to/dir")"
    exit 3
fi
if [ -z ${NMPM_CALIB_CONFIG} ]; then
    echo "Param directory is needed (-p "path/to/param_files")"
    exit 3
fi
