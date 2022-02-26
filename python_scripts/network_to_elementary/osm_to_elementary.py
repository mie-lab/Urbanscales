import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import sys
from shapely import geometry
import numpy as np
import geopandas as gpd


def point_in_polygon(poly, point_x_y):
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
        G = ox.graph_from_polygon(polygon, network_type=network_type, custom_filter=custom_filter)
    else:
        print("Error; wrong input \n\n\n")
        sys.exit()

    # impute edge (driving) speeds and calculate edge traversal times
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


if __name__ == "__main__":
    geo = {
        "type": "Polygon",
        "coordinates": [
            [
                [103.63334655761719, 1.3498201887565244],
                [103.61549377441406, 1.3209889104538992],
                [103.60313415527344, 1.26675774823251],
                [103.57978820800781, 1.2379255258722626],
                [103.58253479003906, 1.1692761289802995],
                [103.68759155273436, 1.1555460449854607],
                [103.80020141601562, 1.2001685712337191],
                [103.90869140625, 1.2386120110295982],
                [104.0130615234375, 1.2688171804967088],
                [104.04876708984375, 1.3065731453895093],
                [104.03984069824219, 1.3786511252106772],
                [104.0020751953125, 1.4225833069396439],
                [103.96774291992186, 1.4218968729661605],
                [103.93478393554688, 1.4301340671471028],
                [103.91349792480467, 1.4246426076343077],
                [103.8922119140625, 1.4315069299741268],
                [103.86886596679688, 1.4534726228737347],
                [103.84071350097656, 1.4747516841452304],
                [103.80500793457031, 1.4733788475908092],
                [103.78578186035156, 1.4589640128389818],
                [103.75762939453124, 1.4472947932028308],
                [103.72810363769531, 1.4617097027979826],
                [103.70613098144531, 1.4534726228737347],
                [103.67385864257812, 1.4294476354255539],
                [103.66012573242188, 1.4033630788431182],
                [103.63334655761719, 1.3498201887565244],
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

    for i in range(1000000):
        lat = np.random.rand() * (max_lat - min_lat) + min_lat
        lon = np.random.rand() * (max_lon - min_lon) + min_lon

        if point_in_polygon(poly, [lat, lon]):
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
    plt.show()

    # custom_filter = '["highway"~"motorway|motorway_link|primary"]',
