#!/bin/bash -exu
wafer=$1
path=$2

cd $path
# merge all wafer files
redman_set_operation.py W${wafer} -o ./ -i $(find -name "wafer-Wafer\(${wafer}\).xml" | xargs dirname) || true
# search for fpga-{wafer}-{fpga}.xml files and evaluate fpga numbers
problematic_fpgas=$(find -name "fpga-${wafer}-*.xml")
# if problematic_fpga is empty grep is failing -> no fpga merging needed -> exit
[ -z "$problematic_fpgas" ] && echo "no defect highspeed connections" && exit 0
problematic_fpgas=$(echo "$problematic_fpgas"|grep -Po "(?<=fpga-${wafer}-).*(?=.xml)")

for fpga in $(echo $problematic_fpgas | tr " " "\n" | sort -nu)
do
    redman_set_operation.py W${wafer}F${fpga} -o ./ -i $(find -name "fpga-${wafer}-${fpga}.xml" | xargs dirname)
done
