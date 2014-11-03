#!/usr/bin/zsh

#dry=echo

measure() {

	voh_dac=$1
	vol_dac=$2

	#date=2014-09-18

	extrasuffix=${run}_vohdac_${voh_dac}_voldac_${vol_dac}

	$dry srun -p wafer defects.py 288 --extrasuffix $extrasuffix --wafer 1 --ana 2>&1 | tee log_${extrasuffix}

}

run=run_7_date_`date -I`

for voh_dac in `seq 0 10 100`
do
	for vol_dac in `seq 0 10 40`
	do
		if [[ $voh_dac -ge $vol_dac ]]
		then
			echo "continue"
			continue
		fi
		echo "at voh DAC ${voh_dac} and vol DAC ${vol_dac}"
		$dry ../l1_voltages.py 10 $voh_dac
		$dry ../l1_voltages.py 9 $vol_dac
		measure $voh_dac $vol_dac
	done
done
