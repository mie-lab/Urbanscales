import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import box
import numpy as np


# Define a point in Mumbai (approximate coordinates of Mumbai city center)
latitude = 19.0760
longitude = 72.8777

# Create a 5x5 km area around this point (approximate conversion of lat/long to km at this latitude)
# 1 degree of latitude ~ 111 km, so 5 km is about 0.045 degrees
half_width_km = 2.5 / 111  # Half the side in degrees

# Creating a bounding box
bbox = box(longitude - half_width_km, latitude - half_width_km,
           longitude + half_width_km, latitude + half_width_km)

# Create grid function
def create_grid(bbox, xsize, ysize):
    xmin, ymin, xmax, ymax = bbox.bounds
    width = xmax - xmin
    height = ymax - ymin
    xsteps = np.arange(xmin, xmax, width / xsize)
    ysteps = np.arange(ymin, ymax, height / ysize)
    grid_boxes = []
    for x in xsteps:
        for y in ysteps:
            grid_boxes.append(box(x, y, x + width / xsize, y + height / ysize))
    return grid_boxes

# Grid sizes in km and their conversion to degrees
sizes_km = [2] # , 1, 0.5]
sizes_deg = [size / 111 for size in sizes_km]  # Convert km to degrees

# Create grids
grids = {size: create_grid(bbox, 5/size, 5/size) for size in sizes_km}


fig, ax = plt.subplots(figsize=(10, 10))

# Convert grids to GeoDataFrames and plot
for size, grid_boxes in grids.items():
    gdf = gpd.GeoDataFrame(geometry=grid_boxes)
    gdf = gdf.set_crs(epsg=4326)  # Set to WGS84
    gdf = gdf.to_crs(epsg=3857)   # Convert to Web Mercator for mapping
    gdf.plot(ax=ax, edgecolor='black', facecolor='none')

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Set axis
ax.set_axis_off()
ax.set_title("Grids over Mumbai: 2x2, 1x1, and 0.5x0.5 sq km")
plt.savefig("Mumbai_demo_4sqkm.png", dpi=300)
plt.show()



sizes_km = [1] # , 1, 0.5]
sizes_deg = [size / 111 for size in sizes_km]  # Convert km to degrees

# Create grids
grids = {size: create_grid(bbox, 5/size, 5/size) for size in sizes_km}


fig, ax = plt.subplots(figsize=(10, 10))

# Convert grids to GeoDataFrames and plot
for size, grid_boxes in grids.items():
    gdf = gpd.GeoDataFrame(geometry=grid_boxes)
    gdf = gdf.set_crs(epsg=4326)  # Set to WGS84
    gdf = gdf.to_crs(epsg=3857)   # Convert to Web Mercator for mapping
    gdf.plot(ax=ax, edgecolor='black', facecolor='none')

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Set axis
ax.set_axis_off()
ax.set_title("Grids over Mumbai: 2x2, 1x1, and 0.5x0.5 sq km")
plt.savefig("Mumbai_demo_1sqkm.png", dpi=300)
plt.show()


sizes_km = [0.5] # , 1, 0.5]
sizes_deg = [size / 111 for size in sizes_km]  # Convert km to degrees

# Create grids
grids = {size: create_grid(bbox, 5/size, 5/size) for size in sizes_km}


fig, ax = plt.subplots(figsize=(10, 10))

# Convert grids to GeoDataFrames and plot
for size, grid_boxes in grids.items():
    gdf = gpd.GeoDataFrame(geometry=grid_boxes)
    gdf = gdf.set_crs(epsg=4326)  # Set to WGS84
    gdf = gdf.to_crs(epsg=3857)   # Convert to Web Mercator for mapping
    gdf.plot(ax=ax, edgecolor='black', facecolor='none')

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Set axis
ax.set_axis_off()
ax.set_title("Grids over Mumbai: 2x2, 1x1, and 0.5x0.5 sq km")
plt.savefig("Mumbai_demo_025sqkm.png", dpi=300)
plt.show()



sizes_km = [2, 1, 0.5]
sizes_deg = [size / 111 for size in sizes_km]  # Convert km to degrees

# Create grids
grids = {size: create_grid(bbox, 5/size, 5/size) for size in sizes_km}


fig, ax = plt.subplots(figsize=(10, 10))

color = {
    2: "tab:blue",
    1: "tab:orange",
    0.5: "tab:red"
}
# Convert grids to GeoDataFrames and plot
for size, grid_boxes in grids.items():
    gdf = gpd.GeoDataFrame(geometry=grid_boxes)
    gdf = gdf.set_crs(epsg=4326)  # Set to WGS84
    gdf = gdf.to_crs(epsg=3857)   # Convert to Web Mercator for mapping
    gdf.plot(ax=ax, edgecolor=color[size], facecolor='none')
    # ax.set_axis_off()
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    # ax.set_title("Grids over Mumbai: 2x2, 1x1, and 0.5x0.5 sq km")
    ax.set_axis_off()
    plt.savefig("Mumbai_demo_" + str(size) + ".png", dpi=300)
    plt.show()

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Set axis
ax.set_axis_off()
ax.set_title("Grids over Mumbai: 2x2, 1x1, and 0.5x0.5 sq km")
plt.savefig("Mumbai_demo_all_combined.png", dpi=300)
plt.show()
