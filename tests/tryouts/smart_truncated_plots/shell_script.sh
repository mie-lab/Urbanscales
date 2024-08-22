#!/bin/bash

# Get a list of all the files with the ".png" extension
files=( $(ls *.png) )

# Loop through each file and extract the last number from the filename
for file in "${files[@]}"
do
    # Extract the number from the filename using a regular expression
    number=$(echo "$file" | sed 's/.*_\([0-9]*\)\.png/\1/')

    # Determine the suffix to use based on the presence of "old" or "new" in the filename
    if [[ "$file" == *"old"* ]]; then
        suffix="_old.png"
    else
        suffix="_new.png"
    fi

    # Rename the file with the new format
    new_name="$number$suffix"
    mv "$file" "$new_name"
done

