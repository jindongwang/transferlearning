REPRUN="../utils/reprun.sh 10"
PYCMD="python3 main.py"

case $1 in
	discr) # CE
		# Results taken from "In search of lost domain generalization" (ICLR'21).
		;;
	cnbb)
		##dataset PACS
		$REPRUN $PYCMD $1 --testdoms 0 --reg_w 1e-4 --reg_s 3e-6 --lr_w 1e-4 --n_iter_w 4
		$REPRUN $PYCMD $1 --testdoms 1 --reg_w 1e-4 --reg_s 3e-6 --lr_w 1e-4 --n_iter_w 4
		$REPRUN $PYCMD $1 --testdoms 2 --reg_w 1e-4 --reg_s 3e-6 --lr_w 1e-4 --n_iter_w 4
		$REPRUN $PYCMD $1 --testdoms 3 --reg_w 1e-4 --reg_s 3e-6 --lr_w 1e-4 --n_iter_w 4
		##dataset VLCS
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 0 --reg_w 1e-4 --reg_s 3e-6 --lr_w 1e-4 --n_iter_w 4
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 1 --reg_w 1e-4 --reg_s 3e-6 --lr_w 1e-4 --n_iter_w 4
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 2 --reg_w 1e-4 --reg_s 3e-6 --lr_w 1e-4 --n_iter_w 4
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 3 --reg_w 1e-4 --reg_s 3e-6 --lr_w 1e-4 --n_iter_w 4
		;;
	svae) # CSGz
		##dataset PACS
		$REPRUN $PYCMD $1 --testdoms 0 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --testdoms 1 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --testdoms 2 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --testdoms 3 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		##dataset VLCS
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 0 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 1 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 2 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 3 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		;;
	svgm) # CSG
		##dataset PACS
		$REPRUN $PYCMD $1 --testdoms 0 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --testdoms 1 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --testdoms 2 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --testdoms 3 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		##dataset VLCS
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 0 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 1 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 2 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 3 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		;;
	svgm-ind) # CSG-ind
		##dataset PACS
		$REPRUN $PYCMD $1 --testdoms 0 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --testdoms 1 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --testdoms 2 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --testdoms 3 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		##dataset VLCS
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 0 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 1 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 2 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 3 --wsup 1. --wlogpi 0. --wl2 5e-4 --wgen 1e-7 --pstd_x 3e-1
		;;
	*)
		echo "unknown argument $1"
		;;
esac

