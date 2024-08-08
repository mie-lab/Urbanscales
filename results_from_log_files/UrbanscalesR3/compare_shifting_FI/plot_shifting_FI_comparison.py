import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define color and alpha values for shifting
colors = {
    "Mumbai": "#E69F00",  # orange
    "Auckland": "#56B4E9",  # sky blue
    "Istanbul": "#009E73",  # bluish green
    "Mexico City": "#F0E442",  # yellow
    "Bogota": "#0072B2",  # blue
    "New York City": "#D55E00",  # vermilion
    "Cape Town": "#CC79A7",  # reddish purple
}
shift_alphas = {'1': 0.6, '3': 1}

# Load data from CSV files
def load_data(file_name, shifting, congestion_type):
    df = pd.read_csv(file_name)
    df['Shifting'] = shifting
    df['Congestion_Type'] = congestion_type
    return df

df1 = load_data("feature_importance_RC_Shifting_1.csv", '1', 'RC')
df2 = load_data("feature_importance_RC_Shifting_3.csv", '3', 'RC')
df3 = load_data("feature_importance_NRC_Shifting_1.csv", '1', 'NRC')
df4 = load_data("feature_importance_NRC_Shifting_3.csv", '3', 'NRC')

# Combine all datasets into one DataFrame
all_data = pd.concat([df1, df2, df3, df4])

# Handle special case for Istanbul
all_data.loc[(all_data['city'] == 'istanbul') & (all_data['scale'] == '75'), 'scale'] = '1 sqkm'
all_data.loc[(all_data['city'] != 'istanbul') & (all_data['scale'] == '50'), 'scale'] = '1 sqkm'

# Pivot the data
pivot_data = all_data.pivot_table(index=['city', 'featurename'], columns=['Congestion_Type', 'Shifting'], values='FI', aggfunc='mean').reset_index()

# Rename cities in title case and correct specific names
pivot_data['city'] = pivot_data['city'].str.title().replace({'Newyorkcity': 'New York City', 'Mexicocity': 'Mexico City', 'Capetown': 'Cape Town'})

# Sort cities alphabetically
pivot_data.sort_values('city', inplace=True)
# Sort features alphabetically
pivot_data.sort_values(['city', 'featurename'], inplace=True)

# Plotting
width = 0.35  # width of the bars
for congestion_type in ['RC', 'NRC']:
    fig, axs = plt.subplots(nrows=len(pivot_data['city'].unique()), figsize=(15, 3 * len(pivot_data['city'].unique())), squeeze=False)
    for i, (city, group) in enumerate(pivot_data.groupby('city')):
        pos = np.arange(len(group['featurename']))
        for shift in ['1', '3']:
            axs[i, 0].bar(pos + (int(shift)-2) * width/2, group[(congestion_type, shift)], width=0.4,
                          color=colors.get(city, 'grey'), alpha=shift_alphas[shift],
                          label=f'Shifting {shift}')
        axs[i, 0].set_title(f'Feature Importance for {city} - {congestion_type}')
        axs[i, 0].set_xticks(pos)
        axs[i, 0].set_xticklabels(group['featurename'], rotation=45, fontsize=12)
        axs[i, 0].set_ylabel('Feature Importance (FI)', fontsize=14)
        axs[i, 0].legend()

    plt.tight_layout()
    plt.savefig(congestion_type + "_tile_shifting.png", dpi=300)
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt





import pandas as pd
import matplotlib.pyplot as plt

# Define function to plot bar chart for specified shifting value
def plot_feature_importance(city_data, shift, congestion_type):
    # Filter data for specified shifting, ensure no missing values, and sort feature names alphabetically
    shift_data = city_data[(congestion_type, shift)].dropna()
    sorted_shift_data = shift_data.sort_index()  # Sort by feature name

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.barh(sorted_shift_data.index.to_list()[::-1], sorted_shift_data.values.tolist()[::-1], color= colors[city_data.name]) # so that the order of features is the same as the FI mean plot

    # Display FI values on the bars
    # for i, value in enumerate(sorted_shift_data.values):
    #     ax.text(value, i, f'{value:.4f}', va='center', ha='left')

    # Set labels and title
    ax.set_xlabel('Feature Importance (FI)', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title(f'FI for {congestion_type} - Shifting {shift} \n {city_data.name}', fontsize=12)
    plt.yticks(range(0, 14), city_data.featurename.to_list(), fontsize=12)  # to get the range right best is to plot without this line then manually copy the range
    plt.xticks([0.01, 0.05, 0.1, 0.15], fontsize=12)
    if congestion_type == "RC":
        plt.xlim(0, 0.1)
    else:
        plt.xlim(0, 0.028)
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.savefig(city_data.name + "_Shifting_FI_"+ congestion_type +"_shifting_"+ str(shift)+ ".png", dpi=300)
    plt.show()

# Example usage with the previously created pivot_data DataFrame
new_york_data = pivot_data[pivot_data['city'] == 'Auckland']
new_york_data.name = "Auckland"  # Set the DataFrame's name property for use in the title
plot_feature_importance(new_york_data, '1', 'NRC')
plot_feature_importance(new_york_data, '3', 'NRC')
plot_feature_importance(new_york_data, '1', 'RC')
plot_feature_importance(new_york_data, '3', 'RC')


