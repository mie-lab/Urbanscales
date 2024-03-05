import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
df_scale_25 = pd.read_csv('./morning_peak_hour_all_cities_scale_25.csv', sep="\t")
df_scale_50 = pd.read_csv('./morning_peak_hour_all_cities_scale_50.csv', sep="\t")
df_scale_100 = pd.read_csv('./morning_peak_hour_all_cities_scale_100.csv', sep="\t")

# Add a 'scale' column to each DataFrame
df_scale_25['scale'] = 25
df_scale_50['scale'] = 50
df_scale_100['scale'] = 100

# Combine the DataFrames
df_combined = pd.concat([df_scale_25, df_scale_50, df_scale_100])

# Normalize the scale for Istanbul
def normalize_scale(row):
    if row['cityname'] == 'Istanbul':
        if row['scale'] == 37:
            return 25
        elif row['scale'] == 75:
            return 50
        elif row['scale'] == 150:
            return 100
    return row['scale']

df_combined['normalized_scale'] = df_combined.apply(normalize_scale, axis=1)

# Filter the data for LR and RF
df_lr = df_combined[df_combined['model'] == 'Linear Regression']
df_rf = df_combined[df_combined['model'] == 'Random Forest']

# Pivot the data for LR
df_pivot_lr = df_lr.pivot_table(index='normalized_scale', columns='cityname', values='GoF_explained_Variance')

# Plot the data for LR
df_pivot_lr.plot(kind='bar', figsize=(10, 6))
plt.title('GoF Based on Explained Variance for LR')
plt.xlabel('Area of each urban tile')
plt.ylabel('GoF (Explained Variance)')
plt.legend(title='City', ncol=3)
plt.ylim(0, 1)
plt.xticks(ticks=[0, 1, 2], labels=[r'4 $km^2$', r'1 $km^2$', r'0.25 $km^2$'], rotation=0)
plt.savefig("Morning_peak_all_cities_GoF_LR")
plt.show()

# Pivot the data for RF
df_pivot_rf = df_rf.pivot_table(index='normalized_scale', columns='cityname', values='GoF_explained_Variance')

# Plot the data for RF
df_pivot_rf.plot(kind='bar', figsize=(10, 6))
plt.title('GoF Based on Explained Variance for RF')
plt.xlabel('Area of each urban tile')
plt.ylabel('GoF (Explained Variance)')
plt.legend(title='City', ncol=3)
plt.ylim(0, 1)
plt.xticks(ticks=[0, 1, 2], labels=[r'4 $km^2$', r'1 $km^2$', r'0.25 $km^2$'], rotation=0)
plt.savefig("Morning_peak_all_cities_GoF_RF")
plt.show()
