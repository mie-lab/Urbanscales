import pandas as pd
from tabulate import tabulate

# Assuming the data is stored in a text file named 'tile_count.csv'
file_path = 'tile_count.csv'

# Read the data
data = []
with open(file_path, 'r') as file:
    for line in file:
        # Skip the initial '#', strip the newline, and split by spaces
        parts = line[1:].strip().split(" ")
        print (parts)
        if "#tiles" in parts[0] and "NON-RECURRENT" in parts[1]:
            city = parts[2].replace("NewYorkCity", "New York City").replace("Capetown", "Cape Town").replace("MexicoCity", "Mexico City")
            scale = int(parts[3])
            tile_count = int(parts[4])
            data.append([city, scale, tile_count])

# Create a DataFrame
df = pd.DataFrame(data, columns=['City', 'Scale', '#Tiles'])

# Sort the DataFrame by City and Scale in descending order
df_sorted = df.sort_values(by=['City', 'Scale'], ascending=[True, True])

# Convert DataFrame to a LaTeX table using tabulate
latex_table = tabulate(df_sorted, tablefmt="latex", headers='keys', showindex=False)

# Print or save the LaTeX table
print(latex_table)
