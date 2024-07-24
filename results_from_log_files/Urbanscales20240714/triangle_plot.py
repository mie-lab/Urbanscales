

import ternary
import matplotlib.pyplot as plt

# Data setup
data = {
    'Mumbai': [9, 90, 1, 20.4],
    'Auckland': [60, 20, 20, 1.4],
    'Istanbul': [29, 65, 6, 15.5],
    'Mexico City': [30, 63, 7, 21.7],
    'Bogota': [22, 69, 9, 10.9],
    'New York City': [22, 57, 21, 8.4],
    'Cape Town': [58, 22, 20, 4.6]
}

colors = {
    'Mumbai': '#E69F00',
    'Auckland': '#56B4E9',
    'Istanbul': '#009E73',
    'Mexico City': '#F0E442',
    'Bogota': '#0072B2',
    'New York City': '#D55E00',
    'Cape Town': '#CC79A7'
}

# Create the figure and tax object for the ternary plot
figure, tax = ternary.figure(scale=100)
tax.set_title("Modal Shares of Transportation", fontsize=20)
tax.boundary(linewidth=2.0)
tax.gridlines(multiple=10, color="blue", alpha=0.3)  # Reduce gridline visibility

# Plot each point
for city, (car, pt, wm, population) in data.items():
    tax.scatter([tuple([car, pt, wm])], label=city, marker='o', s=population * 10, color=colors[city])

# Set axis labels
tax.left_axis_label("Walking/Cycling (%)", fontsize=15)  # Increase font size
tax.right_axis_label("Public Transport (%)", fontsize=15)
tax.bottom_axis_label("Car (%)", fontsize=15)

# Adjust plot to ensure labels are visible
# plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9)
tax.legend()

# Clear default ticks and show the plot
tax.ticks(axis='lbr', linewidth=1, multiple=10)
tax.clear_matplotlib_ticks()
plt.tight_layout()
ternary.plt.show()

