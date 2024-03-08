import matplotlib.pyplot as plt

# Data
import os
import pandas as pd

# List of cities
cities = ['London', 'Capetown', 'NewYorkCity', 'Bogota', 'MexicoCity', 'Auckland', 'Mumbai', 'Zurich', 'Singapore', 'Istanbul']

# Scales
scales = [25, 50]# [25, 50, 100]


for scale in scales:
    # Loop through each city and scale

    # Initialize lists to store the GoF values
    lr_gof = []
    rf_gof = []
    for city in cities:
        city_gof_lr = []
        city_gof_rf = []

        file_path = f'{city}/spatial_tod_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]_scale_{scale}/GOF_spatial_tod_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]_scale_{scale}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Extract GoF values for Linear Regression and Random Forest
            lr_gof_value = df[df['model'] == 'Linear Regression']['GoF_explained_Variance'].mean()
            rf_gof_value = df[df['model'] == 'Random Forest']['GoF_explained_Variance'].mean()
            city_gof_lr.append(lr_gof_value)
            city_gof_rf.append(rf_gof_value)
        else:
            print(f'File not found: {file_path}')
        lr_gof.append(lr_gof_value)
        rf_gof.append(rf_gof_value)

    # Print the resulting lists
    print('cities =', cities)
    print('lr_gof =', lr_gof)
    print('rf_gof =', rf_gof)

    # Combine and sort data
    combined_data = sorted(zip(cities, lr_gof, rf_gof), key=lambda x: x[0])
    sorted_cities, sorted_lr_gof, sorted_rf_gof = zip(*combined_data)

    cities, lr_gof, rf_gof = sorted_cities, sorted_lr_gof, sorted_rf_gof

    # Set up the plot for Linear Regression
    # plt.figure(figsize=(10, 6))
    plt.bar(cities, lr_gof, color=plt.cm.tab10.colors)
    plt.xlabel('City')
    plt.ylabel('GoF Explained Variance')
    plt.title('Linear Regression GoF Explained Variance by City Shift 2 Non recurent')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylim(0, 0.6)
    plt.savefig("LR_GoF_across_cities_shifting_is_2_1sq_km_non_recurrent" +str(scale)+ "spatial_.png")
    plt.show()

    # Set up the plot for Random Forest
    # plt.figure(figsize=(10, 6))
    plt.bar(cities, rf_gof, color=plt.cm.tab10.colors)
    plt.xlabel('City')
    plt.ylabel('GoF Explained Variance')
    plt.title('Random Forest GoF Explained Variance by City Shift 2 Non recurrent')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylim(0, 0.6)
    plt.savefig("RF_GoF_across_cities_shifting_is_2_1sq_km_non_recurrent" +str(scale)+ "spatial.png")
    plt.show()

