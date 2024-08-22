#!/bin/bash -e

# parse input arguments and set variables
. ../parse_input.sh $@

task=prereq

name=$task-w${WAFER}h${HICANN}aout${AOUT}
prefix=$name

# set log name for submission log
if [ -d $OUTDIR ]; then
    logname_submit=$(find "${OUTDIR}" -maxdepth 1 -name "submit_w${WAFER}_h${HICANN}_run*" | sort | tail -n1)
    if [ -z $logname_submit ]; then
        resume_number=0
    else
        resume_number_padded=$(grep -Po 'run\K[^.]+' <<< $logname_submit)
        resume_number=$((10#$resume_number_padded)) #removes leading zeros
        let resume_number+=1
    fi
else
    resume_number=0
fi
printf -v resume_number_padded "%05d" $resume_number # sets resume_number_padded

log_prefix=submit_w${WAFER}_h${HICANN}_run
log=../${log_prefix}${resume_number_padded}.log
exec > >(tee -a $log)
exec 2>&1

if ! mkdir -p $OUTDIR ; then
    echo "cannot create workspace $OUTDIR"
    exit 1
fi

if [[ ! -w $OUTDIR ]]; then
    echo "workspace $OUTDIR is not writable"
    exit 1
fi

# OUTDIR could be created, so we can move the submission log
mv $log ${OUTDIR}/${log_prefix}${resume_number_padded}.log
log=${OUTDIR}/${log_prefix}${resume_number_padded}.log
exec > >(tee -a $log)

if [[ ${LOADEDMODULES} == *localdir* ]]; then
    echo "You are using a localdir cake."
elif [[ ${LOADEDMODULES} == *nmpm_software* ]]; then
    echo "You are using cake from nmpm_software"
else
    echo "No version of cake found. Exiting."
    exit 6
fi

if ! [[ ${LOADEDMODULES} == *slurm-singularity* ]]; then
    echo "Module slurm-singularity should be loaded. Exiting."
    exit 6
fi

if [[ -z $CONTAINER_APP_NMPM_SOFTWARE || -z $CONTAINER_IMAGE_NMPM_SOFTWARE ]]; then
    echo "Both CONTAINER_APP_NMPM_SOFTWARE and CONTAINER_IMAGE_NMPM_SOFTWARE must be set"
    exit 6
fi

echo "$prefix: prereq/submit.sh running at `date -u`"
if [[ ! -z "$SP_DEPENDENCY_ARG" ]]; then
    echo "$prefix: dependencies are $SP_DEPENDENCY_ARG"
fi

if [[ $SP_SIMULATE = "1" ]]; then
    echo "$prefix: this step was skipped"
    echo "TASK: $task"
    echo
    exit 0
fi

export SINGULARITYENV_PREPEND_PATH=$PATH

env

jobid=$(sbatch --parsable --time=00:02:00 -o "${OUTDIR}/slurm_${name}_run${resume_number_padded}.log" -p batch -J ${name} \
       --kill-on-invalid-dep=yes ${SP_DEPENDENCY_ARG} \
       --wrap "singularity exec --app $CONTAINER_APP_NMPM_SOFTWARE $CONTAINER_IMAGE_NMPM_SOFTWARE ./run.sh $SP_ORIGINAL_ARGS")
if [[ -z "${jobid}" ]]; then
    echo "${prefix}: submission failed"
    exit 1
fi
echo "${prefix}: submitted job ${jobid}"
echo "TASK: ${task} ${jobid}"
echo
