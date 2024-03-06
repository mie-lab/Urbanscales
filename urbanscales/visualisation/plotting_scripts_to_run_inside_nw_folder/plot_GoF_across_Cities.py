import matplotlib.pyplot as plt

# Data
cities = ['Auckland', 'Mumbai', 'Bogota', 'Singapore', 'Zurich', 'Capetown', 'MexicoCity', 'NewYorkCity', 'London', 'Istanbul']
lr_gof = [0.4073, 0.1135, 0.3841, 0.4287, 0.2640, 0.3325, 0.4316, 0.4368, 0.0375, 0.3047]
rf_gof = [0.5183, 0.1347, 0.4481, 0.4457, 0.2505, 0.4234, 0.5149, 0.5066, 0.0021, 0.3276]

# Set up the plot for Linear Regression
# plt.figure(figsize=(10, 6))
plt.bar(cities, lr_gof, color=plt.cm.tab10.colors)
plt.xlabel('City')
plt.ylabel('GoF Explained Variance')
plt.title('Linear Regression GoF Explained Variance by City')
plt.xticks(rotation=90)
plt.tight_layout()
plt.ylim(0, 0.6)
plt.savefig("LR_GoF_across_cities_shifting_is_2_1sq_km.png")
plt.show()

# Set up the plot for Random Forest
# plt.figure(figsize=(10, 6))
plt.bar(cities, rf_gof, color=plt.cm.tab10.colors)
plt.xlabel('City')
plt.ylabel('GoF Explained Variance')
plt.title('Random Forest GoF Explained Variance by City')
plt.xticks(rotation=90)
plt.tight_layout()
plt.ylim(0, 0.6)
plt.savefig("RF_GoF_across_cities_shifting_is_2_1sq_km.png")
plt.show()

