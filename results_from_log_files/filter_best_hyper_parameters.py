import os

# Directory where your files are located
directory = './'

# Output CSV file
output_file = 'summary_best_hyper_parameters.csv'

# Keyword to look for
keyword = 'randomforestregressor__max_depth'

# Open the output file
with open(output_file, 'w') as outfile:
    # Write header to the output file
    # outfile.write("Filename,Line\n")

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and ("RECURRENT" in filename or "NONRECURRENT" in filename) and "copy" not in filename:
            # Full path to the current file
            filepath = os.path.join(directory, filename)
            
            # Read all lines from the file
            with open(filepath, 'r') as file:
                lines = file.readlines()
            
            # Loop through lines and search for the keyword
            for i, line in enumerate(lines):
                if keyword in line:
                    # Lines to write, check bounds for lines before and after
                    lines_to_write = []
                    if i > 3:  # Line 3 lines before
                        lines_to_write.append(f"{filename},{lines[i-4]}")
                    lines_to_write.append(f"{filename},{line}")
                    if i+1 < len(lines):  # Line 1 after
                        lines_to_write.append(f"{filename},{lines[i+1]}")
                    if i+2 < len(lines):  # Line 2 after
                        lines_to_write.append(f"{filename},{lines[i+2]}")

                    # Write the collected lines to the output file
                    for l in lines_to_write:
                        outfile.write(l)

print("Filtering complete, results saved to:", output_file)
data = []
with open("summary_best_hyper_parameters.csv") as f:
    for row in f:
        if "scale" in row:
            row_spaced = row.replace(","," ") # replace comma with space (just some dirty log file processing)
            row_spaced = ' '.join(row_spaced.split())  # replace multiple spaces with 1 :); more dirty log
                                                                            # file processing
            city = row_spaced.split(" ")[8]
            scale = row_spaced.split(" ")[9]
            congest_type = "NRC" if "NONRECURRENT" in row_spaced.split(" ")[0] else "RC"
            prefix =  [city, scale, congest_type]

            for row in f:
                row_spaced = row.replace(",", " ")  # replace comma with space (just some dirty log file processing)
                row_spaced = ' '.join(row_spaced.split())
                if "randomforestregressor__max_depth" in row_spaced:
                    max_depth = row_spaced.split(":")[1].replace("}", "")
                    # print ("max_depth", max_depth)
                break
            for row in f:
                row_spaced = row.replace(",", " ")  # replace comma with space (just some dirty log file processing)
                row_spaced = ' '.join(row_spaced.split())
                if "randomforestregressor__max_features" in row_spaced:
                    max_features = row_spaced.split(":")[1].replace("}", "")
                    # print ("max_features", max_features)
                break
            for row in f:
                row_spaced = row.replace(",", " ")  # replace comma with space (just some dirty log file processing)
                row_spaced = ' '.join(row_spaced.split())
                if "randomforestregressor__n_estimators" in row_spaced:
                    n_estimators = row_spaced.split(":")[1].replace("}", "")
                    # print ("n_estimators", n_estimators)
                break
            # data.append([city, scale, congest_type, n_estimators, max_features, max_depth])
            data.append([city, scale, congest_type, n_estimators, max_depth])  # max features removed since they are same for 15 np.log(15) and sqrt(15) have the same floor

            print ("city, scale, congest_type, n_estimators, max_features, max_depth", city, scale, congest_type, n_estimators, max_features, max_depth)

from tabulate import tabulate

# Assuming data is stored in a list of lists
# data = [
#     ["MexicoCity", "40", "RC", "1000", "sqrt", "50"],
#     ["MexicoCity", "40", "NRC", "200", "sqrt", "10"],
#     # Add more data as necessary
# ]

# Convert list of lists to a dictionary grouped by city and scale, distinguishing RC and NRC
data_dict = {}
for item in data:
    key = (item[0], item[1])  # City and scale as key
    if item[2] == "RC":
        data_dict[key] = {**data_dict.get(key, {}), "RC": item[3:]}
    else:
        data_dict[key] = {**data_dict.get(key, {}), "NRC": item[3:]}

# Prepare data for tabulation
table_data = []
for (city, scale), params in data_dict.items():
    rc_params = params.get("RC", ["-", "-", "-"])  # Default if not available
    nrc_params = params.get("NRC", ["-", "-", "-"])  # Default if not available
    row = [city, scale] + rc_params + nrc_params
    table_data.append(row)

# Define headers
headers = ["City", "n~#tiles=(nXn)", "RC Max Depth", "RC Features", "RC Estimators",
           "NRC Max Depth", "NRC Features", "NRC Estimators"]
latex_table = tabulate(table_data, headers=headers, tablefmt="latex")

# Print the LaTeX table
print(latex_table)
