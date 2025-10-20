#!/bin/bash
# Loop through arguments
# USE OR -o config or --output=configfilename WITHOUT .json5
outname=config
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -o|--output)  # Handle both -o and --output
            # Check if the next argument is provided
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for $1"
                exit 1
            fi
            outname="$2"
            shift 2
            ;;
        --output=*)  # Support --output=VALUE
            outname="${1#*=}"
            shift
            ;;
        --) # Stop parsing options
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)  
            # Handle dataset names
            input_dir="../../trained_models_q16/$1"
            if [[ -d "$input_dir" ]]; then
                input_dir_refactored=$(realpath "$input_dir")
                input_datasets+=("$input_dir_refactored")
            else
                echo "Warning: Directory $input_dir does not exist."
            fi
            shift
            ;;
    esac
done


# echo $input_datasets
# exit 1
for ds in ${input_datasets[@]}; do

    for i in $(find $ds -name 'rf_*' -type d | sort | uniq); do
        directory=$(realpath $i)
        dataset=$(basename $(dirname $i))
        pmml_file=$(realpath $(find $i -name '*.pmml'))
        train_data=$(realpath $(find $i -name 'training_set.csv'))
        test_data=$(realpath $(find $i -name 'test_set.csv'))
        data_desc=$(realpath $(find /home/user/shared/Datasets/${dataset} -name 'config.json'))
        
        echo $pmml_file $directory $train_data $test_data $data_desc


        printf "\
    {\n\
        \"model\" : \"%s\",\n\
        \"outdir\" : \"%s/ps_results\",\n\
        \"als\" : {\n\
            \"cache\"    : \"/home/user/lut_catalog.db\",\n\
            \"cut_size\" : 4,\n\
            \"solver\"   : \"btor\"\n\
        },\n\
        \"error\": {\n\
            \"max_loss_perc\" : 5,\n\
            \"training_dataset\": \"%s\",\n\
            \"test_dataset\": \"%s\",\n\
            \"dataset_description\": \"%s\"\n\
        },\n\
        \"optimizer\" : {\n\
            \"archive_hard_limit\"       : 20,\n\
            \"archive_soft_limit\"       : 50,\n\
            \"archive_gamma\"            : 1,\n\
            \"clustering_iterations\"    : 300,\n\
            \"hill_climbing_iterations\" : 10,\n\
            \"initial_temperature\"      : 100,\n\
            \"final_temperature\"        : 1e-1,\n\
            \"cooling_factor\"           : 0.9,\n\
            \"annealing_iterations\"     : 100,\n\
            \"annealing_strength\"       : 1,\n\
            \"early_termination\"        : 5,\n\
            \"multiprocess_enabled\"     : false\n\
        }\n\
    }\n" $pmml_file $directory $train_data $test_data $data_desc > $i/$outname.json5;
    done
done
