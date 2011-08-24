for i in 0 1 2 4 5 6 7 8 
  do 
  for j in 10 20 30 40 60 80 100 
    do
     time=`awk 'BEGIN{onlinetime=0;batchtime=0;batchbegin=0;} {if($1 == "Net" && $2 == "time" && onlinetime == 0) onlinetime=$(NF-1); if($2 == "avg." && $3 == "loss") batchbegin=1; if(batchbegin && $2 < 0.320480) {batchtime=$NF;batchbegin=0}} END{print onlinetime+batchtime}' mapperout"$j"_$i`
     echo -e "$time \c"
  done
  echo ""
done