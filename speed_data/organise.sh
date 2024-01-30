#!/bin/bash

# List of cities
cities=("Singapore" "Zurich" "Mumbai" "Auckland" "Istanbul" "MexicoCity" "Bogota" "NewYorkCity" "Capetown" "London")

# Iterate over each city
for city in "${cities[@]}"; do
    # Create a folder for the current city
    mkdir -p "$city"

    # Find the files matching the city name and move them to the respective folder
    find . -maxdepth 1 -type f \( -name "here_data_speed_*${city}_jf.csv" -o -name "speed-*_${city}_linestring.csv" \) -exec mv {} "$city" \;

    # Rename the files inside the folder
    mv "${city}/here_data_speed_*_${city}_jf.csv" "${city}/jf.csv"
    mv "${city}/speed-*_${city}_linestring.csv" "${city}/segments.csv"
done

