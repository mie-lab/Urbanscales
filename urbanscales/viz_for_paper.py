import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import geopandas as gpd


# Define the location and size of the area
city_name = "Piedmont, California, USA"  # Example city
distance = 3000  # 3 km from the center, giving a 6x6 km area

G = ox.graph_from_point(40.7699002160956, -73.96739201898806)
# Download the graph
# G = ox.graph_from_place(city_name, network_type='drive', retain_all=True, dist=distance)

# Project the graph to UTM (for better distance accuracy)
G = ox.project_graph(G)

# Get the bounding box of the graph
bbox = ox.utils_geo.graph_to_gdfs(G, nodes=False).unary_union.bounds

# Split the bounding box into 1 sq km tiles
xmin, ymin, xmax, ymax = bbox
nrows = int(np.ceil((ymax-ymin)/1000))
ncols = int(np.ceil((xmax-xmin)/1000))
grid_cells = []

for i in range(ncols):
    for j in range(nrows):
        grid_cells.append(box(xmin + i*1000, ymin + j*1000, xmin + (i+1)*1000, ymin + (j+1)*1000))

grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=G.graph['crs'])

# Placeholder function for feature extraction from a tile
def extract_features(G, tile):
    subgraph = ox.truncate.truncate_graph_bbox(G, tile.bounds[1], tile.bounds[3], tile.bounds[0], tile.bounds[2])
    basic_stats = ox.basic_stats(subgraph)
    features = {
        'n': basic_stats.get('n', np.nan),
        'm': basic_stats.get('m', np.nan),
        'k_avg': basic_stats.get('k_avg', np.nan),
        'edge_length_avg': basic_stats.get('edge_length_avg', np.nan),
        'streets_per_node_avg': basic_stats.get('streets_per_node_avg', np.nan),
        'circuity_avg': basic_stats.get('circuity_avg', np.nan),
        'total_crossings': basic_stats.get('total_crossings', np.nan)
    }
    return features

# Extract features for all tiles
features_list = []
for tile in grid.geometry:
    features = extract_features(G, tile)
    features_list.append(features)

# Identify the min and max tiles for each feature
extreme_tiles = {}
for feature_name in features_list[0].keys():
    feature_values = [f[feature_name] for f in features_list]
    min_index = feature_values.index(min(feature_values))
    max_index = feature_values.index(max(feature_values))
    
    extreme_tiles[feature_name] = {
        'min': {'tile': grid.geometry.iloc[min_index], 'value': min(feature_values)},
        'max': {'tile': grid.geometry.iloc[max_index], 'value': max(feature_values)}
    }

# Plot the extreme tiles for each feature
def plot_extreme_tiles(feature_name, low_tile, high_tile, low_value, high_value):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot lowest value tile
    ax[0].set_title(f'Lowest {feature_name}: {low_value}')
    low_subgraph = ox.truncate.truncate_graph_bbox(G, low_tile.bounds[1], low_tile.bounds[3], low_tile.bounds[0], low_tile.bounds[2])
    ox.plot_graph(low_subgraph, ax=ax[0], show=False, close=False)
    
    # Plot highest value tile
    ax[1].set_title(f'Highest {feature_name}: {high_value}')
    high_subgraph = ox.truncate.truncate_graph_bbox(G, high_tile.bounds[1], high_tile.bounds[3], high_tile.bounds[0], high_tile.bounds[2])
    ox.plot_graph(high_subgraph, ax=ax[1], show=False, close=False)
    
    plt.show()

# Plotting all features
for feature_name, tiles in extreme_tiles.items():
    plot_extreme_tiles(
        feature_name,
        tiles['min']['tile'], tiles['max']['tile'],
        tiles['min']['value'], tiles['max']['value']
    )


