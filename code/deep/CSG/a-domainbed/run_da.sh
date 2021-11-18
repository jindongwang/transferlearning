REPRUN="../utils/reprun.sh 10"
PYCMD="python3 main.py"

case $1 in
	dann)
		# Results taken from "In search of lost domain generalization" (ICLR'21).
		;;
	cdan)
		# Results taken from "In search of lost domain generalization" (ICLR'21).
		;;
    dan)
		##dataset PACS
		$REPRUN $PYCMD $1 --testdoms 0 --wl2 5e-4 --wsup 1.   --wda 1e-2
		$REPRUN $PYCMD $1 --testdoms 1 --wl2 5e-4 --wsup 1.   --wda 1e-2
		$REPRUN $PYCMD $1 --testdoms 2 --wl2 5e-4 --wsup 1.   --wda 1e-2
		$REPRUN $PYCMD $1 --testdoms 3 --wl2 5e-4 --wsup 1.   --wda 1e-2
		##dataset VLCS
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 0 --wl2 5e-4 --wsup 1.   --wda 1e-2
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 1 --wl2 5e-4 --wsup 1.   --wda 1e-2
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 2 --wl2 5e-4 --wsup 1.   --wda 1e-2
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 3 --wl2 5e-4 --wsup 1.   --wda 1e-2
		;;
	mdd)
		##dataset PACS
		$REPRUN $PYCMD $1 --testdoms 0 --wl2 5e-4 --wsup 1.   --wda 1e-2
		$REPRUN $PYCMD $1 --testdoms 1 --wl2 5e-4 --wsup 1.   --wda 1e-2
		$REPRUN $PYCMD $1 --testdoms 2 --wl2 5e-4 --wsup 1.   --wda 1e-2
		$REPRUN $PYCMD $1 --testdoms 3 --wl2 5e-4 --wsup 1.   --wda 1e-2
		##dataset VLCS
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 0 --wl2 5e-4 --wsup 1.   --wda 1e-2
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 1 --wl2 5e-4 --wsup 1.   --wda 1e-2
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 2 --wl2 5e-4 --wsup 1.   --wda 1e-2
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 3 --wl2 5e-4 --wsup 1.   --wda 1e-2
		;;
	bnm)
		##dataset PACS
		$REPRUN $PYCMD $1 --testdoms 0 --wl2 5e-4 --wsup 1.   --wda 1.
		$REPRUN $PYCMD $1 --testdoms 1 --wl2 5e-4 --wsup 1.   --wda 1.
		$REPRUN $PYCMD $1 --testdoms 2 --wl2 5e-4 --wsup 1.   --wda 1.
		$REPRUN $PYCMD $1 --testdoms 3 --wl2 5e-4 --wsup 1.   --wda 1.
		##dataset VLCS
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 0 --wl2 5e-4 --wsup 1.   --wda 1.
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 1 --wl2 5e-4 --wsup 1.   --wda 1.
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 2 --wl2 5e-4 --wsup 1.   --wda 1.
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 3 --wl2 5e-4 --wsup 1.   --wda 1.
		;;
	svae-da) # CSGz-DA
		##dataset PACS
		$REPRUN $PYCMD $1 --testdoms 0 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		$REPRUN $PYCMD $1 --testdoms 1 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		$REPRUN $PYCMD $1 --testdoms 2 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		$REPRUN $PYCMD $1 --testdoms 3 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		##dataset VLCS
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 0 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 1 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 2 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 3 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		;;
	svgm-da) # CSG-DA
		##dataset PACS
		$REPRUN $PYCMD $1 --testdoms 0 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		$REPRUN $PYCMD $1 --testdoms 1 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		$REPRUN $PYCMD $1 --testdoms 2 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		$REPRUN $PYCMD $1 --testdoms 3 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		##dataset VLCS
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 0 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 1 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 2 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		$REPRUN $PYCMD $1 --dataset VLCS --testdoms 3 --pstd_x 3e-1 --wl2 5e-4 --wsup 1.   --wlogpi 0.   --wgen 1e-8 --wda 1e-8
		;;
	*)
		echo "unknown argument $1"
		;;
esac

