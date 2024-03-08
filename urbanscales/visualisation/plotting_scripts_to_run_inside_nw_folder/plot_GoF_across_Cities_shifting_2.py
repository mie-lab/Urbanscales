import matplotlib.pyplot as plt

# Data
cities = ['London', 'Capetown', 'New York City', 'Bogota', 'Mexico City', 'Auckland', 'Mumbai', 'Zurich', 'Singapore', 'Istanbul']
lr_gof = [0.037524755019416926, 0.3325340385637176, 0.4367984073075131, 0.38409408465145184, 0.43161078220656424, 0.40733457910777215, 0.11347069539488655, 0.2639531829985381, 0.4286995341404303, 0.30469263359891335]
rf_gof = [0.0021205247565831754, 0.42343067244463434, 0.5066406957491573, 0.44810270092145377, 0.5149573197345174, 0.518345569489867, 0.13468572006702592, 0.25047303987495295, 0.4456910756919607, 0.3275740959237174]

# Combine and sort data
combined_data = sorted(zip(cities, lr_gof, rf_gof), key=lambda x: x[0])
sorted_cities, sorted_lr_gof, sorted_rf_gof = zip(*combined_data)

cities, lr_gof, rf_gof = sorted_cities, sorted_lr_gof, sorted_rf_gof

# Set up the plot for Linear Regression
# plt.figure(figsize=(10, 6))
plt.bar(cities, lr_gof, color=plt.cm.tab10.colors)
plt.xlabel('City')
plt.ylabel('GoF Explained Variance')
plt.title('Linear Regression GoF Explained Variance by City Shift 2')
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
plt.title('Random Forest GoF Explained Variance by City Shift 2')
plt.xticks(rotation=90)
plt.tight_layout()
plt.ylim(0, 0.6)
plt.savefig("RF_GoF_across_cities_shifting_is_2_1sq_km.png")
plt.show()

