#!/bin/bash
res_dir=$1
name=$2
hunvec_params=`echo $@ | cut -f3- -d' '`
python hunvec/seqtag/trainer.py ${hunvec_params} ${res_dir}/${name}.model > ${res_dir}/${name}.log
echo ${hunvec_params} > ${res_dir}/${name}.params
