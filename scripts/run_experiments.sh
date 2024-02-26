#!/bin/bash

my_array=("l27b-meta" "mist7b-mist" "mixt-4bit")
device="auto"

for model in "${my_array[@]}"; do
    #echo "$model"
    python pipeline_config.py "$model" "$device"
done

#for model in "${my_array[@]}"; do
#    python run_eval.py "$model" "$device"
#done