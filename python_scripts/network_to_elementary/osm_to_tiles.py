import matplotlib.patches
import osmnx as ox
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import sys
from shapely import geometry
import numpy as np
import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Point, LineString


def split_poly_to_bb(poly: geometry.Polygon, n, plotting_enabled=False):
    """
    :param poly: shapely polygon
    :param n: ise used to create a list of bounding boxes; total
           number of such boxes = (n X (aspect_ratio * n) ); scaled_n is calculated in this function
    :return:
    """
    min_lon, min_lat, max_lon, max_lat = poly.bounds

    bbox_list = []
    vertical = calculate_ground_distance(min_lat, min_lon, max_lat, min_lon)
    horizontal = calculate_ground_distance(min_lat, min_lon, min_lat, max_lon)
    print("vertical ", vertical // 1000, " km")
    print("horizontal ", horizontal // 1000, " km")
    aspect_ratio = vertical / horizontal
    print("Aspect ratio ", aspect_ratio)

    delta_x = (max_lat - min_lat) / n
    delta_y = (max_lon - min_lon) / (n / aspect_ratio)
    for i in list(np.linspace(min_lat, max_lat, n, endpoint=False)):
        for j in list(np.linspace(min_lon, max_lon, int(n / aspect_ratio) + 1, endpoint=False)):
            bbox_list.append((i, j, i + delta_x, j + delta_y))
    if plotting_enabled:

        for bbox in bbox_list:
            lat1, lon1, lat2, lon2 = bbox
            centre_lon = 0.5 * (lon1 + lon2)
            centre_lat = 0.5 * (lat1 + lat2)
            plt.scatter(centre_lon, centre_lat, s=0.3, color="red")

            # plot rectangle
            if is_bounding_box_in_polygon(poly, bbox):
                color = "green"
            else:
                color = "red"
            plt.gca().add_patch(
                matplotlib.patches.Rectangle((lon1, lat1), lon2 - lon1, lat2 - lat1, lw=0.8, alpha=0.5, color=color)
            )

        plt.xlim([min_lon, max_lon])
        plt.ylim([min_lat, max_lat])
        plt.xlabel("latitude")
        plt.ylabel("longitude")
        plt.savefig("output_images/network_graphs/bbox_inside_polygoin.png", dpi=400)
        plt.show(block=False)

    return bbox_list


def is_bounding_box_in_polygon(poly, bb):
    """

    :param poly:
    :param point_b (2 points -> list of length 4):
    :return: True/False
    """
    top_left = [bb[0], bb[1]]
    bottom_right = [bb[2], bb[3]]
    return is_point_in_polygon(poly, top_left) and is_point_in_polygon(poly, bottom_right)


def is_point_in_bounding_box(lat, lon, bb):
    lat_min, lon_min = [bb[0], bb[1]]
    lat_max, lon_max = [bb[2], bb[3]]

    # this won't work when dealing with boundary conditions such as the poles/180 degree lon
    return lat_min < lat < lat_max and lon_min < lon < lon_max


def is_point_in_polygon(poly, point_x_y):
    """

    :param poly:
    :param point_x_y:
    :return: True/False
    """

    point = geometry.Point(point_x_y[1], point_x_y[0])
    return poly.contains(point)


def fetch_road_network_from_osm_database(
    named_location=None,
    lat=None,
    lon=None,
    polygon=None,
    dist=1000,
    network_type="drive",
    custom_filter=None,
    plotting_with_road_names=False,
    plotting_enabled=False,
    speed_and_travel_time_needed=True,
    empty_allowed=False,
):
    """
    Args:
      named_location: named location if we are using this instead of lat lon (Default value = None)
      lat: latitude of rctangle centre (Default value = None)
      lon: longitude of rectangle centre (Default value = None)
      polygon: polygon of handmade boundaries (Default value = None)
      dist: distance (not 100% what are the dimensions; visualise to make sure we have what we need) (Default value = 1000)
      network_type: drive/walking etc.. refer to osmnx readme@ https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.graph.graph_from_bbox (Default value = "drive")
      custom_filter: highway"] @https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.graph.graph_from_bbox
      plotting_with_road_names: as the name suggests
    Returns:
        G: Saves two graphs and returns one graph G, before splitting segments into cells split/diverge/merge transforms (Default value = None)
    """

    ox.config(use_cache=True, log_console=True)

    # download street network data from OSM and construct a MultiDiGraph model
    if lat != None and lon != None:
        G = ox.graph_from_point((lat, lon), dist=dist, network_type=network_type)
    elif named_location != None:
        G = ox.graph_from_address(
            address=named_location, dist=dist, network_type=network_type, custom_filter='["highway"~"motorway"]'
        )
    elif polygon != None:
        try:
            G = ox.graph_from_polygon(polygon, network_type=network_type, custom_filter=custom_filter)
        except:
            if not empty_allowed:
                print("OSM returned empty json")
                sys.exit()
            else:
                return "Empty_graph"
    else:
        print("Error; wrong input \n\n\n")
        sys.exit()

    # impute edge (driving) speeds and calculate edge traversal times
    if speed_and_travel_time_needed:
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        G = ox.distance.add_edge_lengths(G)

    # The nodes carry the default ids from osmnx, we convert them to sequential node numbers
    G = nx.convert_node_labels_to_integers(G, first_label=1)

    # we collect the coordinates (x,y) of the nodes manually
    pos_nodes = {}
    for u in G.nodes:
        pos_nodes[u] = (G.nodes[u]["x"], G.nodes[u]["y"])

    if plotting_enabled:
        ox.plot_graph(
            G,
            bgcolor="k",
            node_size=0.1,
            edge_linewidth=0.5,
            edge_color="#333333",
            save=True,
            filepath="output_images/network_graphs/original_network.png",
            dpi=1000,
            close=True,
            show=False,
        )

        nx.draw_networkx_nodes(G, pos_nodes, node_size=2)
        nx.draw(G, pos_nodes, connectionstyle="arc3, rad = 0.1", with_labels=True)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("output_images/network_graphs/original_network_as_networkx_with_directions.png", dpi=300)
        plt.show(block=False)
        plt.close()

    # Plotting osm map with road names (labels misaligned)
    if plotting_with_road_names:
        fig, ax = ox.plot_graph(G, bgcolor="k", edge_linewidth=3, node_size=0, show=False, close=False)
        for _, edge in ox.graph_to_gdfs(G, nodes=False).fillna("").iterrows():
            c = edge["geometry"].centroid
            text = edge["name"]
            ax.annotate(text, (c.x, c.y), c="w", fontsize=3, color="green")
        plt.tight_layout()
        plt.savefig("output_images/network_graphs/original_network_as_networkx_with_roadnames.png", dpi=300)
        plt.show(block=False)
        plt.close()

    return G


def calculate_ground_distance(lat_1, lon_1, lat_2, lon_2):
    """

    :param lat_1:
    :param lon_1:
    :param lat_2:
    :param lon_2:
    :return:
    """
    line_string = LineString([Point(lon_1, lat_1), Point(lon_2, lat_2)])
    geod = Geod(ellps="WGS84")
    return float(f"{geod.geometry_length(line_string):.3f}")


def test_distance():
    # Zurich to Lugano
    # should be around 156 km using https://www.nhc.noaa.gov/gccalc.shtml
    ground_truth = 156 * 1000
    calculated = calculate_ground_distance(47.3769, 8.5417, 46.0037, 8.9511)

    # difference of 3 % tolerated
    percentage_diff = abs(ground_truth - calculated) / ground_truth * 100
    print("Percentage diff in distance: ", percentage_diff, " %")
    assert percentage_diff < 3


if __name__ == "__main__":
    geo = {
        "type": "Polygon",
        "coordinates": [
            [
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
        ],
    }
    poly = Polygon([tuple(l) for l in geo["coordinates"][0]])

    # G_proj = osm.project_graph(G)
    # fig, ax = osm.plot_graph(G_proj)
    # , "trunk","trunk_link", "motorway_link","primary","secondary"]
    # custom_filter=["motorway", "motorway_link","motorway_junction","highway"],

    G_OSM = fetch_road_network_from_osm_database(
        polygon=poly, network_type="drive", custom_filter='["highway"~"motorway|motorway_link|primary"]'
    )

    # test_point in polygon
    min_lat = 1.14
    max_lat = 1.47
    min_lon = 103.5
    max_lon = 104.15

    lat_list_green = []
    lon_list_green = []
    lat_list_red = []
    lon_list_red = []

    for i in range(10000):
        lat = np.random.rand() * (max_lat - min_lat) + min_lat
        lon = np.random.rand() * (max_lon - min_lon) + min_lon

        if is_point_in_polygon(poly, [lat, lon]):
            lat_list_green.append(lat)
            lon_list_green.append(lon)
        else:
            lat_list_red.append(lat)
            lon_list_red.append(lon)

    plt.scatter(lon_list_green, lat_list_green, color="green", s=0.1)
    plt.scatter(lon_list_red, lat_list_red, color="red", s=0.1)
    plt.savefig("output_images/network_graphs/points_outside.png", dpi=300)
    plt.show(block=False)

    p = gpd.GeoSeries(poly)
    p.plot()
    plt.savefig("output_images/network_graphs/polygon_sg.png", dpi=300)
    plt.show()

    split_poly_to_bb(poly, 25, plotting_enabled=True)

    test_distance()
