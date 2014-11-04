#!/usr/bin/zsh

#dry=echo

measure() {

	voh_dac=$1
	vol_dac=$2
	freq=$3
	bkgisi=$4

	extrasuffix=${run}_vohdac_${voh_dac}_voldac_${vol_dac}

	$dry srun -p wafer defects.py 288 --extrasuffix $extrasuffix --wafer 1 --freq $freq --bkgisi $bkgisi --ana 2>&1 | tee log_${extrasuffix}

}

run=run_10_date_`date -I`

best_voh_dac=0
best_vol_dac=50

for bkgisi in `seq 5000 2000 15000`
do
	for freq in 125
	do
		for voh_dac in $best_voh_dac # run 8 `seq 0 2 6`
		do
			for vol_dac in $best_vol_dac # run 8`seq 40 -5 0`
			do
				if [[ $voh_dac -ge $vol_dac ]]
				then
					echo "continue"
					continue
				fi
				echo "at voh DAC ${voh_dac} and vol DAC ${vol_dac}"
				$dry ../l1_voltages.py 10 $voh_dac
				$dry ../l1_voltages.py 9 $vol_dac
				measure $voh_dac $vol_dac $freq $bkgisi
			done
		done
	done
done
