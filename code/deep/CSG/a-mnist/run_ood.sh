REPRUN="../utils/reprun.sh 10"
PYCMD="python3 main.py"
TRAIN="--traindom train01_1.0_0.0_randn_5.0_1.0.pt"
TEST="--testdoms test_01.pt test01_0.5_0.5_randn_0.0_2.0.pt"

case $1 in
	discr) # CE
		$REPRUN $PYCMD $1 $TRAIN $TEST --discrstru lite-1.5x --optim RMSprop --lr 1e-3 --wl2 1e-5
		;;
	cnbb)
		$REPRUN $PYCMD $1 $TRAIN $TEST --discrstru lite-1.5x --optim RMSprop --lr 1e-3 --wl2 1e-5 --reg_w 1e-4 --reg_s 3e-6 --lr_w 1e-3 --n_iter_w 4
		;;
	svae) # CSGz
		$REPRUN $PYCMD $1 $TRAIN $TEST --discrstru lite --optim RMSprop --lr 1e-3 --wl2 1e-5 --mu_s .5 --sig_s .5 --pstd_x 3e-2 --qstd_s=-1.  --wgen 1e-4 --wsup 1. --mvn_prior 1
		;;
	svgm) # CSG
		$REPRUN $PYCMD $1 $TRAIN $TEST --discrstru lite --optim RMSprop --lr 1e-3 --wl2 1e-5 --mu_s .5 --sig_s .5 --mu_v .5 --sig_v .5 --corr_sv .9 --pstd_x 3e-2 --qstd_s=-1.  --qstd_v=-1.  --wgen 1e-4 --wsup 1. --mvn_prior 1
		;;
	svgm-ind) # CSG-ind
		$REPRUN $PYCMD $1 $TRAIN $TEST --discrstru lite --optim RMSprop --lr 1e-3 --wl2 1e-5 --mu_s .5 --sig_s .5 --mu_v .5 --sig_v .5 --corr_sv .9 --pstd_x 3e-2 --qstd_s=-1.  --qstd_v=-1.  --wgen 1e-4 --wsup 1. --mvn_prior 1
		;;
	*)
		echo "unknown argument $1"
		;;
esac

