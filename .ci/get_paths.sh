#!/bin/bash
# Select the last produced calibration/evaluation path searching for file results.h5
# In case of calibration, also allow to search for backends folder.
if [[ $# -lt 2 ]]; then
    exit 1
fi

cal_eval=$1
basePath=$2

if [[ $cal_eval = "cal" ]]; then
    if [[ $# -lt 3 ]]; then
        folder_name_prefix="calibration_f"
    else
        echo -e "$(find "$basePath" -name backends | \
            sort | tail -n1 | tr -d '[:space:]')"
        exit 0
    fi
elif [[ $cal_eval = "eval" ]]; then
    folder_name_prefix="evaluation_f"
else
    exit 1
fi

file=$(find "$basePath" -name "results.h5" | \
    grep "$folder_name_prefix" | sort | tail -n1 | tr -d '[:space:]')
if [[ -z $file ]]; then
    exit 1
fi
filePath=$(dirname "$file")
echo "$filePath"
