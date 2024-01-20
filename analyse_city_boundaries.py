"""
[out:json][timeout:50];
(
  // Istanbul
  {{geocodeArea:Istanbul}}->.istanbul;
  relation["boundary"="administrative"]["admin_level"="6"](area.istanbul);

  // Mumbai
  {{geocodeArea:Mumbai Suburban}}->.mumbaisub;
  relation["boundary"="administrative"]["admin_level"="5"](area.mumbaisub);

    // Mumbai
  {{geocodeArea:Mumbai}}->.mumbaiur;
  relation["boundary"="administrative"]["admin_level"="10"](area.mumbaiur);

  // Mexico City
  {{geocodeArea:Mexico City}}->.mexicocity;
  relation["boundary"="administrative"]["admin_level"="6"](area.mexicocity);

  // Singapore
  {{geocodeArea:Singapore}}->.singapore;
  relation["boundary"="administrative"]["admin_level"="6"](area.singapore);

  // Zurich
  {{geocodeArea:Zurich}}->.zurich;
  relation["boundary"="administrative"]["admin_level"="6"](area.zurich);

  // Capetown
  {{geocodeArea:Stad Kaapstad}}->.capetown;
  relation["boundary"="administrative"]["admin_level"="6"](area.capetown);

  // New York City
   {{geocodeArea:New York City}}->.newyorkcity;
  relation["boundary"="administrative"]["admin_level"="6"](area.newyorkcity);

  // Bogota
  {{geocodeArea:Bogota}}->.bogota;
  relation["boundary"="administrative"]["admin_level"="6"](area.bogota);

  // London
  {{geocodeArea:London}}->.london;
  relation["boundary"="administrative"]["admin_level"="5"](area.london);

  //Auckland
  {{geocodeArea:Auckland}}->.auckland;
  relation["boundary"="administrative"]["admin_level"="6"](area.auckland);

);

// print results
out body;
>;
out skel qt;


# https://overpass-turbo.eu/
# The lines above work with Overpass turbo EU; note the german name for capetown
# The output is saved as the geojson file: geojson_for_cities.json
# http://tinyurl.com/osm-cities-10-boundaries
"""


import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import box
import numpy as np

# Define the bounding boxes for each city
city_bboxes = {
    "Singapore": [1.51316, 104.135278, 1.130361, 103.566667],
    "Zurich": [47.434666, 8.625441, 47.32022, 8.448006],
    "Mumbai": [19.270177, 72.979731, 18.893957, 72.776333],
    "Auckland": [-36.681247, 174.925937, -36.965932, 174.63532],
    "Istanbul": [41.671, 29.9581, 40.7289, 27.9714],
    "MexicoCity": [19.592757, -98.940303, 19.048237, -99.364924],
    "Bogota": [4.837015, -73.996423, 4.4604, -74.223689],
    "NewYorkCity": [40.916178, -73.700181, 40.477399, -74.25909],
    "Capetown": [-34.462, 18.1107, -33.3852, 19.0926],
    "London": [51.28676, -0.510375, 51.691874, 0.334015],
}

# Loop through each city and plot its bounding box
for city, bbox in city_bboxes.items():
    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate the center of the city bounding box
    center_lat = (bbox[0] + bbox[2]) / 2
    center_lon = (bbox[1] + bbox[3]) / 2

    # Calculate the bounds of the 50km x 50km square
    # Note: 1 degree of latitude is approximately 111km. The calculation for longitude varies and is approximate.
    lat_offset = 50 / 111
    lon_offset = 50 / (111 * np.cos(np.radians(center_lat)))

    square_bounds = [center_lon - lon_offset, center_lat - lat_offset,
                     center_lon + lon_offset, center_lat + lat_offset]

    # Create a rectangle for the square
    square = box(*square_bounds)

    # Convert the square to a GeoDataFrame
    square_gdf = gpd.GeoDataFrame(geometry=[square], crs='EPSG:4326')
    square_gdf = square_gdf.to_crs(epsg=3857)

    # Plot the square
    square_gdf.plot(ax=ax, facecolor='blue', edgecolor='blue', alpha=0.5)

    # Create a GeoDataFrame for the bounding box
    gdf = gpd.GeoDataFrame({'city': [city]},
                           crs="EPSG:4326",
                           geometry=[box(bbox[3], bbox[2], bbox[1], bbox[0])])

    # Convert to Web Mercator for contextily
    gdf = gdf.to_crs(epsg=3857)

    # Plotting
    gdf.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2)

    # Set the bounds for the plot
    minx, miny, maxx, maxy = gdf.geometry.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Set title
    ax.set_title(f'Bounding Box of {city}')
    plt.savefig(f'Bounding Box of {city}.png', dpi=300)
    plt.show(block=False)



# Load your data
import geopandas as gpd
from mpl_toolkits.basemap import Basemap



# Function to convert latitude and longitude to kilometers
def latlon_to_km(lat, lon):
    R = 6371  # Earth radius in kilometers
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    return x, y


# Load your data (assuming it's a GeoDataFrame with 'latitude' and 'longitude' columns)
gdf = gpd.read_file('geojson_for_cities.geojson')

import matplotlib.pyplot as plt


# Load your data

# Extract points from MultiPolygon geometries
points = []
for geom in gdf.geometry:
    if geom.type == 'MultiPolygon':
        for polygon in geom:
            points.extend([(x, y) for x, y in zip(*polygon.exterior.coords.xy)])
    elif geom.type == 'Polygon':
        points.extend([(x, y) for x, y in zip(*geom.exterior.coords.xy)])


# Extracting coordinates from MultiPolygon geometries
lats, lons = [], []
for geom in gdf.geometry:
    if geom.type == 'MultiPolygon':
        for polygon in geom:
            lats.extend([y for x, y in polygon.exterior.coords])
            lons.extend([x for x, y in polygon.exterior.coords])
    elif geom.type == 'Polygon':
        lats.extend([y for x, y in geom.exterior.coords])
        lons.extend([x for x, y in geom.exterior.coords])
# Initialize the Basemap
plt.figure(figsize=(15, 10))
m = Basemap(projection='merc', llcrnrlat=min(lats), urcrnrlat=max(lats),
            llcrnrlon=min(lons), urcrnrlon=max(lons), lat_ts=20, resolution='i')

# Draw coastlines, countries, and states
m.drawcoastlines()
m.drawcountries()
m.drawstates()

# Convert latitudes and longitudes to x and y coordinates on the map
x, y = m(lons, lats)

# Scatter plot
m.scatter(x, y, s=20, color='red', marker='o', alpha=0.5)

plt.title('Scatter Plot of Coordinates with Basemap')
plt.savefig("Scatter_Plot_of_Coordinates_with_Basemap.png", dpi=300)
plt.show(block=False)


import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN


# Load your data
gdf = gpd.read_file('geojson_for_cities.geojson')

# Extract points from MultiPolygon and Polygon geometries
points = []
for geom in gdf.geometry:
    if geom.type == 'MultiPolygon':
        for polygon in geom.geoms:  # Use .geoms for MultiPolygon
            points.extend([(x, y) for x, y in zip(*polygon.exterior.coords.xy)])
    elif geom.type == 'Polygon':
        points.extend([(x, y) for x, y in zip(*geom.exterior.coords.xy)])

# Convert lat-lon points to a NumPy array
points_array = np.array(points)

# DBSCAN clustering
dbscan = DBSCAN(eps=300 / 6371.0, min_samples=5, metric='haversine', algorithm='ball_tree')
clusters = dbscan.fit_predict(np.radians(points_array))


import matplotlib.pyplot as plt
import contextily as ctx
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
import alphashape
import geopandas as gpd
import numpy as np

# Assuming points_array and clusters are already defined as per your previous steps

# Create concave hulls and plot with basemap for each cluster
counter = 0
for cluster_id in set(clusters):
    counter += 1
    if cluster_id != -1:  # Exclude noise points
        cluster_points = [points_array[i] for i in range(len(points_array)) if clusters[i] == cluster_id]
        hull = alphashape.alphashape(MultiPoint(cluster_points), 0.95)

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=3857)  # Convert to Web Mercator for contextily

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(ax=ax, edgecolor='k', alpha=0.3)

        # Add basemap
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # Set title and axis
        ax.set_title(f'Cluster {cluster_id}')
        ax.axis('off')
        plt.savefig("Cluster " + str(counter) + ".png", dpi=300)
        plt.show(block=False)
