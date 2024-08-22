import networkx as nx
import matplotlib.pyplot as plt

# Create a random graph
G = nx.gnm_random_graph(20, 30, seed=42)

# Calculate streets_per_node_counts
streets_per_node = dict(G.degree())
streets_per_node_counts = {}

for degree in streets_per_node.values():
    streets_per_node_counts[degree] = streets_per_node_counts.get(degree, 0) + 1

# Calculate k_avg
k_avg = sum(streets_per_node.values()) / len(streets_per_node)

# Draw the graph
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title('Random Graph')
plt.show(block=False)

# Display the calculated values
print("streets_per_node_counts:", streets_per_node_counts)
print("k_avg:", k_avg)

