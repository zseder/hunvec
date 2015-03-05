#!/bin/bash
args=("$@")
res_dir=`echo ${args[0]}`
name=`echo ${args[1]}`
hunvec_params=`echo $@|cut -f3- -d' '`
python hunvec/seqtag/trainer.py ${hunvec_params} ${res_dir}/${name}.model > ${res_dir}/${name}.log
echo ${hunvec_params} > ${res_dir}/${name}.params
