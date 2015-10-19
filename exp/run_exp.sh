#!/bin/bash
res_dir=$1
name=$2
hunvec_params=`echo $@ | cut -f3- -d' '`
if [ -f ${res_dir}/${name}.model ];
then 
	echo "file ${res_dir}/${name}.model exists"
fi
date +"%m/%d/%Y %H:%M:%S" > ${res_dir}/${name}.log
nice python hunvec/seqtag/trainer.py ${hunvec_params} ${res_dir}/${name}.model >> ${res_dir}/${name}.log 2> ${res_dir}/${name}.err  
date +"%m/%d/%Y %H:%M:%S" >> ${res_dir}/${name}.log
echo ${hunvec_params} > ${res_dir}/${name}.params
