#!/bin/bash

# Inputs
input_datasets=(dry_bean statlog_segment)
mr_orders=(11)
tree_num=20
fraction=0
ranking_procedure=("pertree_acc_heu" "pertree_margin_heu")
outdir="mrq16_exp"
ncpus=8

tool='/home/user/pyALS-RF-dbg/pyals-rf'
for i in $(seq 1 1); do
    for mr_order in ${mr_orders[@]}; do
        for rk in ${ranking_procedure[@]}; do
            for name in ${input_datasets[@]}; do # For each input dataset, i.e. avila, spambase 
                pruning_directory=/home/user/shared/$outdir/${rk}/${name}/rf_${tree_num}/mr_${mr_order}/cfg_${i}
                report_directory=/home/user/shared/$outdir/${rk}/${name}/rf_${tree_num}
                config_director=/home/user/shared/trained_models_test/${name}/rf_${tree_num}/config.json5
                mkdir -p $pruning_directory
                mkdir -p $report_directory
                if [[ "$fraction" -eq "0" ]]; then
                    # Use the default none value of the fraction input parameter.
                    $tool mr_heu -c $config_director -p $pruning_directory -d $report_directory -m $mr_order -r $rk -j $ncpus
                else
                    $tool mr_heu -c $config_director -p $pruning_directory -d $report_directory -f $fraction -m $mr_order -r $rk  -j $ncpus
                fi
            done
        done
    done
done


