import sys

import matplotlib.pyplot as plt
from smartprint import smartprint as sprint
import osmnx as ox
import os
from shapely.geometry import Polygon
import pickle
import config

os.makedirs(os.path.join(config.intermediate_files_path, "multiple_cities"), exist_ok=True)
os.makedirs(os.path.join(config.intermediate_files_path, "multiple_cities"), exist_ok=True)
os.makedirs(os.path.join(config.intermediate_files_path, "multiple_cities"), exist_ok=True)
os.makedirs(os.path.join(config.intermediate_files_path + "multiple_cities", "raw_graphs_from_OSM"), exist_ok=True)
os.remove(os.path.join(config.intermediate_files_path, "multiple_cities", "node_count.txt"), exist_ok=True)

# do not delete this
os.makedirs(os.path.join("output_folder", "multiple_cities", "raw_graphs_from_OSM_pickles"))


def check_available_cities():
    ox.config(use_cache=True, log_console=True)
    city_list = [
        "Auckland",
        "Bogota",
        "Cape Town",
        "Istanbul",
        "London",
        "Mexico City",
        "Mumbai",
        "New York City",
        "Singapore",
        "Zurich",
    ]
    for city in city_list:
        filename = config.intermediate_files_path + "multiple_cities/raw_graphs_from_OSM_pickles/" + city + ".pickle"
        import time

        if os.path.isfile(filename):
            starttime = time.time()
            with open(filename, "rb") as f:
                G = pickle.load(f)
            print(time.time() - starttime, city)
        else:
            try:
                # '["highway"~"motorway|motorway_link"]'
                G = ox.graph_from_place(city, network_type="drive", custom_filter=None)
            except:
                print("Error in city: ", city)
                print("Trying out query using polygon: ")
                poly = get_poly_from_list_of_coords(get_poly(city))
                try:
                    G = ox.graph_from_polygon(poly, network_type="drive", custom_filter=None)
                except:
                    print("Error in city: ", city)
                    continue

                # sys.exit(0)

            with open(filename, "wb") as f:
                pickle.dump(G, f, protocol=4)

        sprint(city, len(list(G.nodes)), len(list(G.edges)))

        with open(config.intermediate_files_path + "multiple_cities/node_count.txt", "a") as f:
            sprint(city, len(list(G.nodes)), len(list(G.edges)), file=f)

        if config.plotting_enabled:
            fig, ax = ox.plot_graph(G, edge_linewidth=0.1, node_size=1)
            fig.savefig(
                config.intermediate_files_path + "multiple_cities/raw_graphs_from_OSM/" + city + ".png", dpi=300
            )
            plt.show(block=False)


def get_poly_from_list_of_coords(polygon):

    coords = polygon["coords"]
    if polygon["format"] == "lat_lon":
        coords_invert = []
        for lat_lon in coords:
            lat, lon = lat_lon
            coords_invert.append([lon, lat])
        coords = coords_invert
    elif polygon["format"] == "lon_lat":
        do_nothing = True
    else:
        print("Something wrong in format")
        sys.exit(0)

    geo = {"type": "Polygon", "coordinates": [coords]}
    return Polygon([tuple(l) for l in geo["coordinates"][0]])


def get_poly(city):
    if city == "Cape Town":
        points = [
            [19.06402587890625, -33.62719851659248],
            [19.153976440429688, -33.89321737944087],
            [19.139556884765625, -33.94905609818091],
            [18.906784057617188, -34.15954545771159],
            [18.834686279296875, -34.17204456998344],
            [18.785934448242188, -34.092473191457664],
            [18.696670532226562, -34.07996230865872],
            [18.629379272460938, -34.07768740409025],
            [18.56689453125, -34.087923993423324],
            [18.51333618164062, -34.098727939581174],
            [18.472824096679688, -34.110667538758996],
            [18.360214233398438, -34.06744957739346],
            [18.356781005859375, -34.054366109559815],
            [18.34579467773437, -34.06289903506279],
            [18.331375122070312, -34.06688077296647],
            [18.294296264648438, -34.04355650412745],
            [18.301849365234375, -34.021933159447485],
            [18.395233154296875, -33.87953701355922],
            [18.457717895507812, -33.89606717952166],
            [18.474884033203125, -33.87839688404626],
            [18.445358276367188, -33.79626754190988],
            [18.52020263671875, -33.77343983379775],
            [18.595733642578125, -33.83734207824605],
            [18.654098510742188, -33.784854448488915],
            [18.7646484375, -33.81908916394127],
            [18.939056396484375, -33.77343983379775],
            [18.958969116210938, -33.62319623534053],
            [19.06402587890625, -33.62719851659248],
        ]
    return {"format": "lon_lat", "coords": points}


if __name__ == "__main__":
    check_available_cities()
