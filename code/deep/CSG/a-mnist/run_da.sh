REPRUN="../utils/reprun.sh 10"
PYCMD="python3 main.py"
TRAIN="--traindom train01_1.0_0.0_randn_5.0_1.0.pt"
TEST="--testdoms test_01.pt test01_0.5_0.5_randn_0.0_2.0.pt"

case $1 in
	dann)
		$REPRUN $PYCMD $1 $TRAIN $TEST --discrstru lite-1.5x --optim RMSprop --lr 3e-4 --wl2 1e-5 --wda 1e-4
		;;
	cdan)
		$REPRUN $PYCMD $1 $TRAIN $TEST --discrstru lite-1.5x --optim RMSprop --lr 3e-4 --wl2 1e-5 --wda 1e-6
		;;
    dan)
		$REPRUN $PYCMD $1 $TRAIN $TEST --discrstru lite-1.5x --optim RMSprop --lr 3e-4 --wl2 1e-5 --wda 1e-8
		;;
	mdd)
		$REPRUN $PYCMD $1 $TRAIN $TEST --discrstru lite-1.5x --optim RMSprop --lr 3e-4 --wl2 1e-5 --wda 1e-6
		;;
	bnm)
		$REPRUN $PYCMD $1 $TRAIN $TEST --discrstru lite-1.5x --optim RMSprop --lr 3e-4 --wl2 1e-5 --wda 1e-7
		;;
	svae-da) # CSGz-DA
		$REPRUN $PYCMD $1 $TRAIN $TEST --discrstru lite --optim RMSprop --lr 3e-4 --wl2 1e-5 --wda 1e-4 --mu_s .5 --sig_s .5 --pstd_x 3e-2 --qstd_s=-1.  --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-4 --wsup 1.
		;;
	svgm-da) # CSG-DA
		$REPRUN $PYCMD $1 $TRAIN $TEST --discrstru lite --optim RMSprop --lr 3e-4 --wl2 1e-5 --wda 1e-4 --mu_s .5 --sig_s .5 --mu_v .5 --sig_v .5 --corr_sv .9 --pstd_x 3e-2 --qstd_s=-1.  --qstd_v=-1.  --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-4 --wsup 1.
		;;
	*)
		echo "unknown argument $1"
		;;
esac

