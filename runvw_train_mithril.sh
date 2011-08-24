out_directory=$1
master=$2
onlinepasses=$3
batchpasses=$4
regularization=$5
mapper=`printenv mapred_task_id | cut -d "_" -f 5`
echo $mapper > /dev/stderr
rm -f temp.cache
echo 'Starting training' > /dev/stderr
echo $1 > /dev/stderr
echo $regularization > /dev/stderr
#./vw -b 24 --cache_file temp.cache --passes 20 --regularization=1 -d /dev/stdin -f tempmodel --master_location $2 --bfgs --mem 5 
#bfgscmd="./vw -b 24 --cache_file temp.cache --passes 10 --regularization=1 --loss_function=logistic -d /dev/stdin -f model --master_location $2 --bfgs --mem 5" 
#./vw -b 24 --cache_file temp.cache --passes 1 -d /dev/stdin -i tempmodel -t
gdcmd="./vw -b 24 --cache_file temp.cache --passes $onlinepasses -q up -q ua -q us -q pa -q ps -q as --regularization=$regularization --adaptive --exact_adaptive_norm -d /dev/stdin -f onlinemodel  --master_location $master --loss_function=logistic --save_per_round --learning_rate=20" 
if [ "$onlinepasses" -gt 0 ]
    then
    bfgscmd="./vw -b 24 --cache_file temp.cache --bfgs --mem 5 --passes $batchpasses -q up -q ua -q us -q pa -q ps -q as --regularization=$regularization --master_location $master -f model -i onlinemodel --loss_function=logistic --save_per_round"
    cgcmd="./vw -b 24 --cache_file temp.cache --bfgs --mem 0 --hessian_on --passes $batchpasses -q up -q ua -q us -q pa -q ps -q as --regularization=1 --master_location $master -f model -i onlinemodel --loss_function=logistic"
else
    bfgscmd="./vw -b 24 --cache_file temp.cache --bfgs --mem 5 --passes $batchpasses -q up -q ua -q us -q pa -q ps -q as --regularization=$regularization --master_location $master -f model --loss_function=logistic --save_per_round"
    cgcmd="./vw -b 24 --cache_file temp.cache --bfgs --mem 0 --hessian_on --passes $batchpasses -q up -q ua -q us -q pa -q ps -q as --regularization=1 --master_location $master -f model --loss_function=logistic"
fi
echo $onlinepasses $batchpasses > /dev/stderr
if [ "$mapper" == '000000' ]
then
    if [ "$onlinepasses" -gt 0 ]
	then
	echo online > /dev/stderr
	$gdcmd > mapperout 2>&1
    fi
    if [ "$batchpasses" -gt 0 ]
	then
	echo batch > /dev/stderr
	$bfgscmd >> mapperout 2>&1
    fi
    outfile=$out_directory/model
    #outfile=$out_directory/tempmodel
    mapperfile=$out_directory/mapperout
    found=`hadoop fs -lsr | grep $out_directory | grep mapperout`
    if [ "$found" != "" ]
    then
	hadoop fs -rmr $mapperfile
    fi
    found=`hadoop fs -lsr | grep $out_directory | grep model`
    if [ "$found" != "" ]
    then
	hadoop fs -rmr $out_directory/*model*
    fi
    echo $outfile > /dev/stderr
    for file in `ls -1 *model*`
      do
      hadoop fs -put $file "$out_directory"/"$file"
    done
    hadoop fs -put mapperout $mapperfile
else
    if [ "$onlinepasses" -gt 0 ]
	then
	echo online > /dev/stderr
	$gdcmd
    fi
    if [ "$batchpasses" -gt 0 ]
	then
	echo batch > /dev/stderr
	$bfgscmd
    fi
fi