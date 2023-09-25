#!/bin/bash

# Navigate to the parent directory
cd ../

# Array of scales
scales=(25 50 100)

# Iterate over each scale value
for scale in "${scales[@]}"; do
    for file in feature_importance_*_${scale}x${scale}_all_values.csv; do
        # Extract the base name without the 'all_values' part
        base_name=$(basename "$file" _all_values.csv)
        
        # Extract header
        header=$(head -1 "$file")
        
        # Create train file with lines containing "train"
        echo "$header" > "${base_name}_train.csv"
        grep "train" "$file" >> "${base_name}_train.csv"
        
        # Create val file with lines not containing "train"
        echo "$header" > "${base_name}_val.csv"
        grep -v "train" "$file" | tail -n +2 >> "${base_name}_val.csv"
    done
done
