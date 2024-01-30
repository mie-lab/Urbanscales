# Loop through each file matching the pattern "normalised-*-PDP-*.png"
for file in unnormalised-*-PDP-*.png; do
    # Extract the last part of the filename using parameter expansion
    foldername="${file##*-PDP-}"
    foldername="${foldername%.png}"

    # If a directory with the folder name doesn't exist, create it
    [ ! -d "$foldername" ] && mkdir "$foldername"

    # Move the file into the corresponding folder
    mv "$file" "$foldername"/
done

