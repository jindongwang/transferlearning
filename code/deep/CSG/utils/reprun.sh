n=$1
command=$2
shift
shift
for i in $(seq 1 $n)
do
	$command $*
done

