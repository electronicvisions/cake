#!/bin/bash -e

# parse input arguments and set variables
. ../parse_input.sh $SP_ORIGINAL_ARGS

if [[ $@ == calib* ]]; then
    CAL_EVAL="calib"
elif [[ $@ == eval* ]]; then
    CAL_EVAL="eval"
else
    echo "Can not determine CAL_EVAL with $@"
    exit 4
fi

echo "CAL_EVAL: " $CAL_EVAL

task=plot_$CAL_EVAL
name=${task}-w${WAFER}h${HICANN}aout${AOUT}

# set log name for submission log
log_prefix=submit_w${WAFER}_h${HICANN}_run
resume_number=$(grep -Po 'run\K[^.]+' <<< $(find "${OUTDIR}" -maxdepth 1 -name "${log_prefix}*" | sort | tail -n1))
log=$OUTDIR/${log_prefix}${resume_number}.log
exec > >(tee -a $log)
exec 2>&1

echo "$name: plot/sbatch.sh running at `date -u`"
if [ ! -z "$SP_DEPENDENCY_ARG" ]; then
    echo "$name: dependencies are $SP_DEPENDENCY_ARG"
fi

if [[ $SP_SIMULATE = "1" ]] || [[ $SP_SKIP = "1" ]]; then
    echo "$name: this step was skipped"
    echo "TASK: $task"
    echo
    exit 0
fi

export SINGULARITYENV_PREPEND_PATH=$PATH

jobid=$(sbatch --parsable --time=00:20:00 --nice=25 -o "$OUTDIR/slurm_${name}_run${resume_number}.log" -p calib -J $name \
        --kill-on-invalid-dep=yes $SP_DEPENDENCY_ARG \
        --wrap "singularity exec --app $CONTAINER_APP_NMPM_SOFTWARE $CONTAINER_IMAGE_NMPM_SOFTWARE ./run.sh $SP_ORIGINAL_ARGS -t=$CAL_EVAL")
if [[ -z "$jobid" ]]; then
echo "$name: submission failed"
    exit 1
    fi
echo "$name: submitted job $jobid"
echo "TASK: $task $jobid"
echo
