REPRUN="../utils/reprun.sh 10"
PYCMD="python3 main.py"

case $1 in
	discr) # CE
		# Results taken from "Conditional adversarial domain adaptation" (NeurIPS'18)
		;;
	cnbb)
		$REPRUN $PYCMD $1 --traindom c --testdoms p   --n_bat 32 --dim_btnk 1024 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --reg_w 1e-6 --reg_s 3e-6 --lr_w 1e-4 --n_iter_w 4 --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom i --testdoms p   --n_bat 32 --dim_btnk 1024 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --reg_w 1e-6 --reg_s 3e-6 --lr_w 1e-4 --n_iter_w 4 --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom p --testdoms c i --n_bat 32 --dim_btnk 1024 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --reg_w 1e-6 --reg_s 3e-6 --lr_w 1e-4 --n_iter_w 4 --n_epk 20 --eval_interval 1
		;;
	svae) # CSGz
		$REPRUN $PYCMD $1 --traindom c --testdoms p   --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v   0 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --sig_s 3e+1 --sig_v   0. --corr_sv 0. --pstd_x 1e-1 --qstd_s=-1.  --qstd_v=0.   --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-7 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom i --testdoms p   --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v   0 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --sig_s 3e+1 --sig_v   0. --corr_sv 0. --pstd_x 1e-1 --qstd_s=-1.  --qstd_v=0.   --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-7 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom p --testdoms c i --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v   0 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --sig_s 3e+1 --sig_v   0. --corr_sv 0. --pstd_x 1e-1 --qstd_s=-1.  --qstd_v=0.   --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-7 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		;;
	svgm) # CSG
		$REPRUN $PYCMD $1 --traindom c --testdoms p   --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v 128 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --sig_s 3e+1 --sig_v 3e+1 --corr_sv .7 --pstd_x 1e-1 --qstd_s=-1.  --qstd_v=-1.  --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-8 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom i --testdoms p   --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v 128 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --sig_s 3e+1 --sig_v 3e+1 --corr_sv .7 --pstd_x 1e-1 --qstd_s=-1.  --qstd_v=-1.  --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-8 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom p --testdoms c i --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v 128 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --sig_s 3e+1 --sig_v 3e+1 --corr_sv .7 --pstd_x 1e-1 --qstd_s=-1.  --qstd_v=-1.  --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-8 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		;;
	svgm-ind) # CSG-ind
		$REPRUN $PYCMD $1 --traindom c --testdoms p   --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v 128 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --sig_s 3e+1 --sig_v 3e+1 --corr_sv .7 --pstd_x 1e-1 --qstd_s=-1.  --qstd_v=-1.  --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-8 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom i --testdoms p   --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v 128 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --sig_s 3e+1 --sig_v 3e+1 --corr_sv .7 --pstd_x 1e-1 --qstd_s=-1.  --qstd_v=-1.  --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-8 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		$REPRUN $PYCMD $1 --traindom p --testdoms c i --n_bat 32 --dims_bb2bn      --dim_btnk 1024 --dim_v 128 --vbranch 0 --dims_bn2v      --dims_bn2s 1024 --dim_s  256 --dims_s2y     --dim_feat 128 --optim SGD --lr 1e-3 --wl2 5e-4 --lr_expo .75 --lr_wdatum 6.25e-6 --sig_s 3e+1 --sig_v 3e+1 --corr_sv .7 --pstd_x 1e-1 --qstd_s=-1.  --qstd_v=-1.  --tgt_mvn_prior 1 --src_mvn_prior 1 --wgen 1e-8 --genstru DCGANpretr --n_epk 20 --eval_interval 1
		;;
	*)
		echo "unknown argument $1"
		;;
esac

