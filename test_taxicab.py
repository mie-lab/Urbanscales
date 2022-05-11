import networkx
import osmnx as ox
import taxicab as tc
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString


list_of_OD = [
    (1.3288, 103.86708, 1.32578, 103.87121),
    (1.33844, 103.86226, 1.33335, 103.8625),
    (1.42558, 103.78923, 1.42417, 103.7922),
    (1.35925, 103.7017, 1.35692, 103.69964),
    (1.31381, 103.87501, 1.31356, 103.87504),
    (1.31407, 103.87498, 1.31381, 103.87501),
]


for orig_dest in list_of_OD:
    orig = (orig_dest[0], orig_dest[1])
    dest = (orig_dest[2], orig_dest[3])

    # orig = (1.34294, 103.74631)
    # dest = (1.34499, 103.74022)

    G = ox.graph_from_point(orig, dist=500, network_type="drive")
    G = ox.speed.add_edge_speeds(G)
    G = ox.speed.add_edge_travel_times(G)

    try:
        route = tc.distance.shortest_path(G, orig, dest)
        x_main, y_main = [], []
        x_first, y_first = [], []
        x_last, y_last = [], []
        if route[1]:
            for u, v in zip(route[1][:-1], route[1][1:]):
                # if there are parallel edges, select the shortest in length
                data = min(G.get_edge_data(u, v).values(), key=lambda d: d["length"])
                if "geometry" in data:
                    # if geometry attribute exists, add all its coords to list
                    xs, ys = data["geometry"].xy
                    x_main.extend(xs)
                    y_main.extend(ys)
                else:
                    # otherwise, the edge is a straight line from node to node
                    x_main.extend((G.nodes[u]["x"], G.nodes[v]["x"]))
                    y_main.extend((G.nodes[u]["y"], G.nodes[v]["y"]))

        # process partial edge first one
        if route[2]:
            x_first, y_first = zip(*route[2].coords)

        # process partial edge last one
        if route[3]:
            x_last, y_last = zip(*route[3].coords)

        lon_list = x_main + list(x_first) + list(x_last)
        lat_list = y_main + list(y_first) + list(y_last)

        XYcoord = (np.column_stack((lat_list, lon_list)).flatten()).tolist()
        # at this point, we should get lat, lon, lat, lon, ......  and so on

        assert len(XYcoord) % 2 == 0
        linestring = LineString(list(zip(XYcoord[0::2], XYcoord[1::2])))

        print(XYcoord)

    except networkx.exception.NetworkXNoPath:
        print("Route not found! Error: networkx.exception.NetworkXNoPath between nodes")
        continue

    fig, ax = tc.plot.plot_graph_route(G, route, node_size=30, show=False, close=False, figsize=(10, 10))
    padding = 0.001
    ax.scatter(orig[1], orig[0], c="lime", s=200, label="orig", marker="x")
    ax.scatter(dest[1], dest[0], c="red", s=200, label="dest", marker="x")
    # ax.set_ylim([min([orig[0], dest[0]]) - padding, max([orig[0], dest[0]]) + padding])
    # ax.set_xlim([min([orig[1], dest[1]]) - padding, max([orig[1], dest[1]]) + padding])
    plt.show()
    plt.savefig(repr(orig_dest) + ".png", dpi=300)
