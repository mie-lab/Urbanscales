import os
import glob


def process_and_copy_files(pattern):
    # List all files matching the pattern
    files = glob.glob(pattern)

    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()

        # Find the line number where the text "nside the model fi" appears for the first time
        target_line_index = next((i for i, line in enumerate(lines) if "nside the model-fi" in line), None)

        # Check if the text was found
        if target_line_index is not None:
            content_to_retain = lines[target_line_index:]

            # Create a new file name with an underscore prefix
            citycount = file[-5]
            new_file = f"{file}"[:-5] + "city" + str(citycount) + ".csv"

            # Write the retained content to the new file
            with open(new_file, 'w') as nf:
                nf.writelines(content_to_retain)
            print(f"Processed and created: {new_file}")
        else:
            print(f"Skipped {file} as no target text was found.")


# Process files matching the described patterns
for i in range(2, 9):  # From 2 to 8
    process_and_copy_files(f"NONRECURRENT{i}.csv")
    process_and_copy_files(f"RECURRENT{i}.csv")

