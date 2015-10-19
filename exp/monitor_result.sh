model=${1}.model
dataset=$2
res_file=$3
time=$4
all_measure=$5

logfile=${1}.log
sleep $time
lastepoch=0


for i in $(seq 1 $all_measure); 
do  

    save_line=`tac $logfile | grep -m1 -n 'model done'| cut -f1 -d:`
    if [ "$save_line" != '' ];then
        epoch=`tac $logfile |tail -n+${save_line} | grep -m1 'Epochs seen'| cut -f3 -d' '` 
        if [ "$epoch" != "$lastepoch" ];then
            date >> $res_file
            echo $epoch >> $res_file
            python hunvec/seqtag/eval.py $dataset $model --sets test --fscore --precision >> $res_file;
            lastepoch=$epoch
        fi
    fi
    sleep $time;
done
