n=0
while [ $n -le 300 ]
do 
	n=$(( $n + 1 ))
	python3 wakanda_run_weyl $1 $2 $3
	
	sleep 3600
	#echo $n
	
done

