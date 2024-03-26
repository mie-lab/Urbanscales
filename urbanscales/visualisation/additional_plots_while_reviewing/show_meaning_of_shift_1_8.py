import geopy.distance
import folium

# Define the center of NYC
center_nyc = (40.7128, -74.0060)

# Define shifts in kilometers and their corresponding styles
shifts = {
    0: {'shift': (0, 0), 'color': 'black', 'weight': 10, 'opacity': 0.01},  # No shift :)
    1: {'shift': (2/3, 0), 'color': 'red', 'weight': 4, 'opacity': 0.01},    # North
    2: {'shift': (-2/3, 0), 'color': 'red', 'weight': 4, 'opacity': 0.01},  # South
    3: {'shift': (0, 2/3), 'color': 'red', 'weight': 4, 'opacity': 0.01},     # East
    4: {'shift': (0, -2/3), 'color': 'red', 'weight': 4, 'opacity': 0.01},  # West
    # 5: {'shift': (2/7, 0), 'color': 'blue', 'weight': 6, 'opacity': 0.01},   # North
    # 6: {'shift': (-2/7, 0), 'color': 'blue', 'weight': 6, 'opacity': 0.01},  # South
    # 7: {'shift': (0, 2/7), 'color': 'blue', 'weight': 6, 'opacity': 0.01},     # East
    # 8: {'shift': (0, -2/7), 'color': 'blue', 'weight': 6, 'opacity': 0.01},  # West


}

# Function to calculate bounding box for a given center and side length
def calculate_bbox(center, side_km):
    half_side = side_km / 2 ** 0.5
    ne_corner = geopy.distance.distance(kilometers=half_side).destination(center, bearing=45)
    sw_corner = geopy.distance.distance(kilometers=half_side).destination(center, bearing=225)
    return [sw_corner.latitude, sw_corner.longitude, ne_corner.latitude, ne_corner.longitude]

# Create a folium map centered at NYC
nyc_map = folium.Map(location=center_nyc, zoom_start=14)

# Plot each shifted tile with specified styles and label it directly on the map
for shift, info in shifts.items():
    lat_shift, lon_shift = info['shift']
    shifted_center = geopy.distance.distance(kilometers=lat_shift).destination(center_nyc, bearing=0)
    shifted_center = geopy.distance.distance(kilometers=lon_shift).destination(shifted_center, bearing=90)

    print (shifted_center)
    bbox = calculate_bbox(shifted_center, 1)  # 1 sqkm side length
    folium.Rectangle(bounds=[[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
                     color=info['color'],
                     weight=info['weight'],
                     fill=True,
                     fill_opacity=info['opacity']).add_to(nyc_map)
    # Calculate the adjusted location for the label
    label_location = geopy.distance.distance(kilometers=0.).destination(((bbox[2] + bbox[0])/2, (bbox[1] + bbox[3]) / 2), bearing=0)  # Move 0.1 km north
    print ("Nishant")
    # Add the marker with increased font size and adjusted position
    folium.map.Marker(location=[label_location.latitude, label_location.longitude],
                      icon=folium.DivIcon(html=f'<div style="color: {info["color"]}; font-size: 36px;">{shift}</div>')).add_to(nyc_map)


    # Display the map
    nyc_map.save('nyc_shifted_tiles.html')
