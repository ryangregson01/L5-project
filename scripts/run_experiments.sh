#!/bin/bash

# Define an array
my_array=("l27b-meta" "mist7b-mist")
device="auto"

# Print each element of the array
for model in "${my_array[@]}"; do
    #echo "$model"
    python pipeline_config.py "$model" "$device"
done
