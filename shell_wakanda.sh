n=0
while [ $n -le 300 ]
do 
	n=$(( $n + 1 ))
	
	
	python3 wakanda_run_weyl.py $1 $2 $3 >/dev/null &
	
	sleep 3600
	#echo $n
	
done

