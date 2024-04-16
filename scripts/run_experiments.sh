#!/bin/bash

my_array=("mixt-noreply" "l27b-noreply" "mist-noreply")
device="auto"

for model in "${my_array[@]}"; do
    #echo "$model"
    python pipeline_config.py "$model" "$device"
done

#for model in "${my_array[@]}"; do
#    python run_eval.py "$model" "$device"
#done
