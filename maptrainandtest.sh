out_directory=$1
train_directory=$2
test_directory=$3
nmappers=$4
realmappers=$5
onlinepasses=$6
batchpasses=$7
regularization=$8
echo $regularization
hadoop fs -rmr $out_directory > /dev/null 2>&1; 
hadoop fs -rmr "$out_directory"_tmp > /dev/null 2>&1; 
total=`hadoop fs -ls $train_directory | cut -d " " -f 7 | awk 'BEGIN{sum = 0} {if(NF > 0) sum += $1;} END{print sum;}'`
echo $total
mapsize=`expr $total / $nmappers`
maprem=`expr $total % $nmappers`
mapsize=`expr $mapsize + $maprem`
mapsize=`expr $mapsize + 100`
echo $mapsize
killall allreduce_master
./allreduce_master $realmappers 2
master=`hostname`
mapcommand="runvw_train.sh "$out_directory"_tmp $master $onlinepasses $batchpasses $regularization"
echo $mapcommand
hadoop jar $HADOOP_HOME/hadoop-streaming.jar -Dmapred.job.queue.name=unfunded -Dmapred.min.split.size=$mapsize -Dmapred.reduce.tasks=0 -Dmapred.job.map.memory.mb=3000 -Dmapred.child.java.opts="-Xmx100m" -Dmapred.task.timeout=600000000 -input $train_directory  -output "$out_directory"_tmp -file vw -file runvw_train.sh -mapper "$mapcommand" -reducer NONE
testcommand="runvw_test.sh $out_directory "$out_directory"_tmp $master"
echo $testcommand
killall allreduce_master
./allreduce_master $realmappers `expr $onlinepasses + $batchpasses + 2`
hadoop jar $HADOOP_HOME/hadoop-streaming.jar -Dmapred.job.queue.name=unfunded -Dmapred.min.split.size=$mapsize -Dmapred.reduce.tasks=0 -Dmapred.child.java.opts="-Xmx100m" -Dmapred.task.timeout=600000000 -files hdfs://axoniteblue-nn1.blue.ygrid.yahoo.com:8020/user/alekh/"$out_directory"_tmp -input $test_directory  -output $out_directory -file vw -file runvw_test.sh -mapper "$testcommand" -reducer NONE