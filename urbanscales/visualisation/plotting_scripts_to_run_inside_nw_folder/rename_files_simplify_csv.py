import os
import shutil

def rename_file(old_name):
    # Define the renaming rules
    renaming_rules = {
        'tod_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]': 'tod_all_day',
        'tod_[9, 10]': 'tod_9AM',
        'tod_[16, 17, 18, 19, 20]': 'tod_evening_peak',
        'tod_[6, 7, 8, 9, 10]': 'tod_morning_peak'
    }

    # Apply the renaming rules
    for old_pattern, new_pattern in renaming_rules.items():
        if old_pattern in old_name:
            return old_name.replace(old_pattern, new_pattern)
    
    # If no rule applies, return the original name
    return old_name

def copy_and_rename_pngs(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv') and 'tod' in file:
                # Construct the full file path
                file_path = os.path.join(root, file)

                # Rename the file according to the rules
                new_file_name = rename_file(file)

                # Construct the new file path
                new_file_path = os.path.join(root, new_file_name)

                # Copy and rename the file
                shutil.copyfile(file_path, new_file_path)
                print(f"Copied and renamed: {file} to {new_file_name}")

if __name__ == "__main__":
    # Run the function on the current directory
    copy_and_rename_pngs('.')

