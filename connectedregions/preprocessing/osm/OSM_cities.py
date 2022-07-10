import sys

import matplotlib.pyplot as plt
from smartprint import smartprint as sprint
import osmnx as ox
import os
from shapely.geometry import Polygon
import pickle
import config
import time

os.makedirs(os.path.join(config.intermediate_files_path, "multiple_cities"), exist_ok=True)
os.makedirs(os.path.join(config.intermediate_files_path, "multiple_cities"), exist_ok=True)
os.makedirs(os.path.join(config.intermediate_files_path, "multiple_cities"), exist_ok=True)
os.makedirs(os.path.join(config.intermediate_files_path + "multiple_cities", "raw_graphs_from_OSM"), exist_ok=True)

fname = os.path.join(config.intermediate_files_path, "multiple_cities", "node_count.txt")
if os.path.exists(fname):
    os.remove(fname)


# do not delete this
os.makedirs(os.path.join("output_folder", "multiple_cities", "raw_graphs_from_OSM_pickles"), exist_ok=True)


def download_OSM_for_cities():

    ox.config(use_cache=True, log_console=True)

    city_list = config.city_list

    for city in city_list:
        filename = os.path.join(
            config.intermediate_files_path, "multiple_cities", "raw_graphs_from_OSM_pickles", city + ".pickle"
        )
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

    elif city == "Singapore":
        points = [
            [103.96078535200013, 1.39109935100015],
            [103.98568769600007, 1.38544342700007],
            [103.99952233200003, 1.38031647300005],
            [104.00342858200003, 1.374172268000066],
            [103.99187259200011, 1.354925848000036],
            [103.97486412900014, 1.334458726000065],
            [103.95435631600009, 1.318101304000052],
            [103.93189537900008, 1.311468817000076],
            [103.90723717500009, 1.308742580000114],
            [103.88770592500003, 1.301255601000136],
            [103.85271243600005, 1.277289130000085],
            [103.84693444100009, 1.271918036000045],
            [103.84408613400012, 1.268500067000034],
            [103.83887780000003, 1.266262111000046],
            [103.82601972700007, 1.264308986000089],
            [103.80160566500007, 1.264797268000081],
            [103.78956139400003, 1.26788971600007],
            [103.78443444100003, 1.273871161000088],
            [103.77588951900009, 1.287583726000108],
            [103.75513756600003, 1.297105210000012],
            [103.73015384200011, 1.302923895000063],
            [103.70875084700003, 1.305243231000119],
            [103.66529381600009, 1.304103908000087],
            [103.6476343110001, 1.308417059000092],
            [103.64039147200003, 1.322251695000091],
            [103.64470462300005, 1.338039455000043],
            [103.67457116000003, 1.38031647300005],
            [103.67888431100005, 1.399237372000073],
            [103.68384850400008, 1.40989817900001],
            [103.69507897200009, 1.421332098000065],
            [103.70834394600013, 1.429388739000089],
            [103.7179468110001, 1.430975653000118],
            [103.73975670700008, 1.428127346000082],
            [103.76221764400009, 1.430975653000118],
            [103.79004967500003, 1.444281317000048],
            [103.80494225400008, 1.448635158000045],
            [103.83155358200003, 1.447088934000092],
            [103.85718834700009, 1.438706773000135],
            [103.93246504000007, 1.401109117000132],
            [103.96078535200013, 1.39109935100015],
        ]

    return {"format": "lon_lat", "coords": points}


if __name__ == "__main__":
    download_OSM_for_cities()
