#!/usr/bin/zsh

#dry=echo

change() {

	#return

	repeat=$1
	direction=$2
	voltage=$3

	echo "changing voltage ${voltage} in direction ${direction}"

	$dry ssh -F /dev/null -x -i ./id_rsa_resetuser resetuser@raspeval-001 -o  PubkeyAuthentication=yes "for i in \`seq ${repeat}\`; do /home/pi/voltages/chVoltage ${direction} ${voltage}; echo changed \$i ${direction} ${voltage}; done"

	$dry sleep 5 # allow voltages to settle

}

measure() {

	voh_step=$1
	vol_step=$2

	date=`date -I`
	#date=2014-09-18

	# TODO: calc date only once

	extrasuffix=${date}_vohstep_${voh_step}_volstep_${vol_step}

	$dry srun -p wafer defects.py 288 --extrasuffix $extrasuffix --wafer 1 --correctoffset 2>&1 | tee log_${extrasuffix}
	#python defects.py 288 --extrasuffix $extrasuffix --wafer 1 --anaonly

}

# start from highest values for VOH and VOL, then step down
echo "resetting voh and vol"
change 150 u 9
change 150 u 10
echo "done"

#ok="1,2 1,3 1,4 2,2 2,3 2,4 3,2 3,3 3,4"

step_size=2

for step_voh in `seq 0 8`
do

	# lower vol first
	change 12 d 9
	change 8 d 9

	for step_vol in `seq 0 8`
	do
		echo "at voh step ${step_voh} and vol step ${step_vol}"
		#if echo $ok | grep "$step_voh,$step_vol" > /dev/null; then
		measure $step_voh $step_vol # measure
		#else
		#	echo "skip"
		#fi
		change $step_size d 9  # decrease vol for next measurement
	done
	change 150 u 9 # reset vol
	change $step_size d 10 # decrease voh for next measurement
done
