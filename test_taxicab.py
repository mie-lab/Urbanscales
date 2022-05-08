import osmnx as ox
import taxicab as tc
import matplotlib.pyplot as plt



list_of_OD = [
    (1.3288,103.86708,1.32578,103.87121),
    (1.33844,103.86226,1.33335,103.8625),
    (1.42558,103.78923,1.42417,103.7922),
    (1.35925,103.7017,1.35692,103.69964),
    (1.31381,103.87501,1.31356,103.87504),
    (1.31407,103.87498,1.31381,103.87501),
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
    except:
        print ("Route not found! ")
        continue

    fig, ax = tc.plot.plot_graph_route(G, route, node_size=30, show=False, close=False, figsize=(10, 10))
    padding = 0.001
    ax.scatter(orig[1], orig[0], c="lime", s=200, label="orig", marker="x")
    ax.scatter(dest[1], dest[0], c="red", s=200, label="dest", marker="x")
    # ax.set_ylim([min([orig[0], dest[0]]) - padding, max([orig[0], dest[0]]) + padding])
    # ax.set_xlim([min([orig[1], dest[1]]) - padding, max([orig[1], dest[1]]) + padding])
    plt.savefig(repr(orig_dest) + ".png", dpi=300)
