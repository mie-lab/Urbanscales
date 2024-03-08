import matplotlib.pyplot as plt

# Data
cities = ['Istanbul', 'London', 'New York City', 'Mexico City', 'Capetown', 'Auckland', 'Bogota', 'Singapore', 'Zurich', 'Mumbai']
lr_gof = [0.06422012872477958, 0.027109982245290154, 0.39264421060433313, 0.25012298083993845, 0.26954639724531254, 0.2513801582785645, 0.24728093171454868, 0.41950376649264315, 0.0596704898480544, 0.04634520895285493]
rf_gof = [-0.06986015291279118, -0.04233958382354322, 0.43946431393236274, 0.3936930595644441, 0.3380625539646588, 0.2891871080324586, 0.18951474692689482, 0.39226241700217873, 0.016551897554981182, -0.058584478970723025]


# Combine and sort data
combined_data = sorted(zip(cities, lr_gof, rf_gof), key=lambda x: x[0])
sorted_cities, sorted_lr_gof, sorted_rf_gof = zip(*combined_data)

cities, lr_gof, rf_gof = sorted_cities, sorted_lr_gof, sorted_rf_gof

# Set up the plot for Linear Regression
# plt.figure(figsize=(10, 6))
plt.bar(cities, lr_gof, color=plt.cm.tab10.colors)
plt.xlabel('City')
plt.ylabel('GoF Explained Variance')
plt.title('Linear Regression GoF Explained Variance by City Shift 3')
plt.xticks(rotation=90)
plt.tight_layout()
plt.ylim(0, 0.6)
plt.savefig("LR_GoF_across_cities_shifting_is_3_1sq_km_spatial.png")
plt.show()

# Set up the plot for Random Forest
# plt.figure(figsize=(10, 6))
plt.bar(cities, rf_gof, color=plt.cm.tab10.colors)
plt.xlabel('City')
plt.ylabel('GoF Explained Variance')
plt.title('Random Forest GoF Explained Variance by City Shift 3')
plt.xticks(rotation=90)
plt.tight_layout()
plt.ylim(0, 0.6)
plt.savefig("RF_GoF_across_cities_shifting_is_3_1sq_km_spatial.png")
plt.show()

