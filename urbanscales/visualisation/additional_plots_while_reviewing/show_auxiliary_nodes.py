import osmnx as ox
import matplotlib.pyplot as plt

def plot_nyc_subgraph():
    # Define the center of the area (Latitude and Longitude of a point in NYC)
    center_point = (41.03040358557639, 29.086623790690076)  # Example: New York City

    # Define the distance from the center to create a 1 sq km area (approximately)
    distance = 500  # meters

    # Create a bounding box around the center point
    north, south, east, west = ox.utils_geo.bbox_from_point(center_point, dist=distance)

    # Download the road network graph for the area
    G = ox.graph_from_bbox(north, south, east, west, network_type='drive')

    # Plot the graph
    fig, ax = ox.plot_graph(G, node_color='red', edge_color='gray', bgcolor='white')

    # Save the plot
    plt.savefig('Istanbul_subgraph.png')

# Run the function
plot_nyc_subgraph()
