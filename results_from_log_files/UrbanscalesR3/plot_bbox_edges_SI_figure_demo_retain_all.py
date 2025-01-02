import osmnx as ox
import matplotlib.pyplot as plt

# Set the bounding box coordinates

# from box number 67 in Auckland scale 100
# north, south, east, west = -37.04402506341895, -37.04853057878709, 174.88254917927574, 174.8769448703508

# north, south, east, west = -37.04402506341895, -37.04853057878709, 174.87134056142585, 174.8657362525009

# box number 63
north, south, east, west = -37.04402506341895, -37.04853057878709, 174.86013194357596, 174.85452763465102

# Download the graph
G = ox.graph_from_bbox(north, south, east, west, network_type='drive', retain_all=True, truncate_by_edge=False, simplify=False)

# Compute basic statistics
stats = ox.basic_stats(G)
print(stats)

# Plot the graph
fig, ax = ox.plot_graph(G)
plt.show()

