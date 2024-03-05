#!/bin/bash

# Define the output file
output_file="concatenated_data.csv"

# Check if the output file already exists and remove it
if [ -f "$output_file" ]; then
    rm "$output_file"
fi

# Loop through all CSV files and append them to the output file
find . -name "*.csv" -print0 | while IFS= read -r -d '' csv_file; do
    # Skip the output file if it's found in the search
    if [[ "$csv_file" != "./$output_file" ]]; then
        cat "$csv_file" >> "$output_file"
    fi
done

echo "Concatenation complete. Output file: $output_file"

