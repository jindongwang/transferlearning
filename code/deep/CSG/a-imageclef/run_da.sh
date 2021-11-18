REPRUN="../utils/reprun.sh 10"
PYCMD="python3 main.py"

case $1 in
	dann)
		# Results taken from "Conditional adversarial domain adaptation" (NeurIPS'18)
		;;
	cdan)
		# Results taken from "Conditional adversarial domain adaptation" (NeurIPS'18)
		;;
    dan)
		# Results taken from "Conditional adversarial domain adaptation" (NeurIPS'18)
		;;
	mdd)
		$REPRUN $PYCMD $1 --traindom c --testdoms p   --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v 128 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --wda 1e-2 --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom i --testdoms p   --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v 128 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --wda 1e-2 --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom p --testdoms c i --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v 128 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --wda 1e-2 --n_epk 20 --eval_interval 1
		;;
	bnm)
		$REPRUN $PYCMD $1 --traindom c --testdoms p   --wl2 5e-4 --wsup 1.   --wda 1.   --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom i --testdoms p   --wl2 5e-4 --wsup 1.   --wda 1.   --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom p --testdoms c i --wl2 5e-4 --wsup 1.   --wda 1.   --n_epk 20 --eval_interval 1
		;;
	svae-da) # CSGz-DA
		$REPRUN $PYCMD $1 --traindom c --testdoms p   --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v   0 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --wda 1e-8 --sig_s 3e+1 --sig_v   0. --corr_sv 0. --pstd_x 1e-1 --qstd_s=-1.  --qstd_v=0.   --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-8 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom i --testdoms p   --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v   0 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --wda 1e-8 --sig_s 3e+1 --sig_v   0. --corr_sv 0. --pstd_x 1e-1 --qstd_s=-1.  --qstd_v=0.   --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-8 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom p --testdoms c i --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v   0 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --wda 1e-8 --sig_s 3e+1 --sig_v   0. --corr_sv 0. --pstd_x 1e-1 --qstd_s=-1.  --qstd_v=0.   --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-8 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		;;
	svgm-da) # CSG-DA
		$REPRUN $PYCMD $1 --traindom c --testdoms p   --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v 128 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --wda 1e-8 --sig_s 3e+1 --sig_v 3e+1 --corr_sv .7 --pstd_x 1e-1 --qstd_s=-1.  --qstd_v=-1.  --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-8 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom i --testdoms p   --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v 128 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --wda 1e-8 --sig_s 3e+1 --sig_v 3e+1 --corr_sv .7 --pstd_x 1e-1 --qstd_s=-1.  --qstd_v=-1.  --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-8 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom p --testdoms c i --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v 128 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 256 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --wda 1e-8 --sig_s 3e+1 --sig_v 3e+1 --corr_sv .7 --pstd_x 3e-1 --qstd_s=-1.  --qstd_v=-1.  --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-8 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		;;
	*)
		echo "unknown argument $1"
		;;
esac

