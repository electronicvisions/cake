#!/bin/bash -e

name=$task-w${WAFER}h${HICANN}aout${AOUT}
prefix=$name

# set log name for submission log
log_prefix=submit_w${WAFER}_h${HICANN}_run
resume_number=$(grep -Po 'run\K[^.]+' <<< $(find "${OUTDIR}" -maxdepth 1 \
                                            -name "${log_prefix}*" | sort -n | tail -n1))
log=$OUTDIR/${log_prefix}${resume_number}.log
exec > >(tee -a $log)
exec 2>&1

echo "$prefix: ${CAL_EVAL}/sbatch.sh running at `date -u`"
if [[ ! -z "$SP_DEPENDENCY_ARG" ]]; then
    echo "$prefix: dependencies are $SP_DEPENDENCY_ARG"
fi

if [[ "${CAL_EVAL}" == "calib" ]]; then
    nicelvl=10
elif [[ "${CAL_EVAL}" == "eval" ]]; then
    nicelvl=2000
else
    echo "CAL_EVAL has to be 'calib' or 'eval', but is ${CAL_EVAL}. Exiting."
fi

if [[ $SP_SIMULATE = "1" ]] || [[ $SP_SKIP = "1" ]]; then
    echo "$prefix: this step was skipped"
    echo
    echo "TASK: $task"
    exit 0
fi

if [[ "${CAL_EVAL}" == "calib" ]] && ! [[ -z "${BACKEND_PATH}" ]]; then
    echo "$prefix: Provided backend path but did not skip calibration. Exiting."
    exit 1
fi

export SINGULARITYENV_PREPEND_PATH=$PATH

# TODO: If only fits should be redone, it is not necessary to allocate the
# wafer
jobid=$(sbatch --parsable --nice="${nicelvl}" -o "$OUTDIR/slurm_${name}_run${resume_number}.log" -p calib \
       --wmod $WAFER --hicann-with-aout ${HICANN}:$AOUT -J $name --kill-on-invalid-dep=yes --time=02:00:00 \
       $SP_DEPENDENCY_ARG --wrap "singularity exec --app $CONTAINER_APP_NMPM_SOFTWARE $CONTAINER_IMAGE_NMPM_SOFTWARE \
       ./run.sh $SP_ORIGINAL_ARGS -t=$CAL_EVAL")
if [[ -z "$jobid" ]]; then
echo "$prefix: submission failed"
    exit 1
    fi
echo "$prefix: submitted job $jobid"
echo "TASK: $task $jobid"
echo
