#!/bin/bash

path=$1
counter=1
while [ -d "${path}-${counter}" ]
do
	counter=$((counter+1))
done
final_path="${path}-${counter}"
mkdir $final_path
chmod 777 $final_path
echo $final_path
