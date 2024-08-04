from geopy.distance import geodesic
from shapely.geometry import Polygon
from shapely.ops import transform
import pyproj

# Define the bounding box (South, North, East, West)
# bbox = [-36.823589499999976, -36.833440384881776, 174.67626514167614, 174.66401177900397]

# N, S, E, W = -37.069861622044975, -36.57731737795502, 175.0320322103899, 174.5292247896101
# N, E, S, W = -36.597979041973296, 175.06749859692312, -37.04853057878709, 174.50706048130448
# N,S,E,W = [-36.86830996406159, -36.87732099479786, 174.9890310488509, 174.977822431001]  # Auckland
N,E,S, W= [41.536727763335755, 29.42207364792899, 40.861416311196756, 28.527917511481668] # Istanbul

bbox = [N, S, E, W]


# Calculate the corners of the bounding box
south, north, east, west = bbox
sw = (south, west)
nw = (north, west)
ne = (north, east)
se = (south, east)

# Calculate geodesic distances between corners
south_side = geodesic(sw, se).kilometers
north_side = geodesic(nw, ne).kilometers
west_side = geodesic(sw, nw).kilometers
east_side = geodesic(se, ne).kilometers

# Print the lengths of each side
print("South Side Length:", south_side, "km")
print("North Side Length:", north_side, "km")
print("West Side Length:", west_side, "km")
print("East Side Length:", east_side, "km")

# Calculate the area using spherical excess formula (approximation)
# Reference Ellipsoid: WGS-84
wgs84 = pyproj.CRS('EPSG:4326')
project = pyproj.Transformer.from_crs(wgs84, wgs84, always_xy=True).transform
# Create a polygon based on the bbox corners
poly = Polygon([sw, nw, ne, se])
# Approximate the area (in sq. km)
area = transform(project, poly).area / 10**6  # converting sq meters to sq km

print("Approximate Geodesic Area:", area, "sq km")

