out_directory=$1
model_directory=$2
master=$3
mapper=`printenv mapred_task_id | cut -d "_" -f 5`
echo $mapper > /dev/stderr
echo $out_directory > /dev/stderr
echo $model_directory > /dev/stderr
#testcmd="./vw -b 24 --cache_file temp.cache --passes 1 -q up -q ua -q us -q pa -q ps -q as -d /dev/stdin --loss_function=logistic --master_location $master -t -i "
testcmd="./vw -b 24 --cache_file temp.cache --passes 1 -d /dev/stdin --loss_function=logistic --master_location $master -t -i "
for file in `ls -1 "$model_directory"/*model*`
  do
  runcmd="$testcmd $file"
  if [ "$mapper" == '000000' ]
      then
      echo $file >> testmapperout
      $runcmd >> testmapperout 2>&1
  else
      echo $file > /dev/stderr
      $runcmd
  fi
done

if [ "$mapper" == '000000' ]
    then
    hadoop fs -copyFromLocal "$model_directory"/mapperout "$out_directory"/mapperout
    hadoop fs -put testmapperout "$out_directory"/testmapperout
    #hadoop fs -put "$model_directory"/model "$out_directory"/model
fi
