#!/bin/bash
# INSERT THE DATASET NAMES BY FOLDERS. 
declare -A datasets
declare -A configs
# Default value for fraction
fraction=0.5  
# Set of input datasets
input_datasets=(statlog_segment dry_bean )

echo $fraction
echo $input_datasets

trees=( 20 )

# IF YOU WANT TO ADD A DS PLEASE ADD CSV and PyALS Cfg to these two lists.

datasets[statlog_segment]='/home/user/shared/Datasets/statlog_segment/segment.dat'
configs[statlog_segment]='/home/user/shared/Datasets/statlog_segment/config.json'

datasets[dry_bean]='/home/user/shared/Datasets/dry_bean/drybean.csv'
configs[dry_bean]='/home/user/shared/Datasets/dry_bean/config.json'


pyalsrf='/home/user/pyALS-RF-dbg/train autotune rf grid'
curr=$PWD
for name in ${input_datasets[@]}; do
    for tree in ${trees[@]}; do
        
        dest_dir=/home/user/shared/trained_models_q16_test/${name}/rf_${tree}
        # echo $name
        # echo $dest_dir
        # echo ${datasets[$name]}
        # echo ${configs[$name]}
        mkdir -p $dest_dir && cd $dest_dir && $pyalsrf ${datasets[$name]} ${configs[$name]} $dest_dir -n ${tree} -j 20 -f $fraction -q16 1 && cd $curr 
    done
done