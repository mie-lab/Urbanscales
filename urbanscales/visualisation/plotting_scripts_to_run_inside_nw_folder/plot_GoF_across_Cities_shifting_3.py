import matplotlib.pyplot as plt

# Data
cities = ['Istanbul', 'London', 'New York City', 'Mexico City', 'Capetown', 'Auckland', 'Bogota', 'Singapore', 'Zurich', 'Mumbai']
lr_gof = [0.2930411039039428, 0.046253906663582915, 0.442847513328383, 0.3148931883912546, 0.3260814455292928, 0.3610697215130931, 0.345297951389448, 0.4611573896510556, 0.22094218043311406, 0.08767441028017538]
rf_gof = [0.3343281710424218, 0.009352948868598911, 0.5348369055420578, 0.45988800036426813, 0.43193019046632336, 0.4461894062545785, 0.36763434436549225, 0.42397579434612137, 0.19655402643579425, 0.09366891503275132]


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
plt.savefig("LR_GoF_across_cities_shifting_is_3_1sq_km.png")
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
plt.savefig("RF_GoF_across_cities_shifting_is_3_1sq_km.png")
plt.show()

