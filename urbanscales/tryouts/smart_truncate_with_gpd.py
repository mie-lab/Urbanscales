import time
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, LineString, Point
import osmnx as ox
import pandas as pd
import shapely
from osmnx import utils_graph
from shapely.geometry import Point
from shapely.geometry import Point, LineString
from shapely.ops import split
from smartprint import smartprint as sprint
import config

N, S, E, W = (
    1.3235381983186159,
    1.319982801681384,
    103.85361309942331,
    103.84833190057668,
)
graph = ox.graph_from_bbox(N, S, E, W, network_type="drive")

# graph = ox.graph_from_point((N, E), dist=500)

nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)
edges = ox.graph_to_gdfs(graph, edges=True, nodes=False)
fig, ax = ox.plot.plot_graph(
    graph,
    ax=None,
    figsize=(10, 10),
    bgcolor="white",
    node_color="red",
    node_size=5,
    node_alpha=None,
    node_edgecolor="none",
    node_zorder=1,
    edge_color="black",
    edge_linewidth=0.1,
    edge_alpha=None,
    show=False,
    close=False,
    save=False,
    bbox=None,
)
W_ = W + (E - W) * 0.8
S_ = S + (N - S) * 0.7
width = (E - W) * 0.07
height = (N - S) * 0.1

rect = plt.Rectangle((W_, S_), width, height, facecolor="green", alpha=0.3, edgecolor=None)
ax.add_patch(rect)
plt.show()

g_truncated = ox.truncate.truncate_graph_bbox(graph, S_ + height, S_, W_ + width, W_, truncate_by_edge=False)
ox.plot_graph(g_truncated)


G = graph
gs_nodes, gs_edges = utils_graph.graph_to_gdfs(G)
sprint(gs_nodes.shape)

old_num_edges = gs_edges.shape[0]
old_num_nodes = gs_nodes.shape[0]

E = W_ + width
W = W_
S = S_
N = S_ + height
bbox_lines = [
    LineString([Point(W, S), Point(E, S)]),
    LineString([Point(E, S), Point(E, N)]),
    LineString([Point(E, N), Point(W, N)]),
    LineString([Point(W, N), Point(W, S)]),
    LineString([Point(W, S), Point(E, S)]),
]

single_bbox_poly_shapely = Polygon([(W, S), (E, S), (E, N), (W, N)])
bbox_poly_series = gpd.GeoSeries(
    [single_bbox_poly_shapely],
)

intersecting_edges_series = gs_edges.intersection(Polygon([(W, S), (E, S), (E, N), (W, N)]))
intersecting_nodes_series = gs_nodes.intersection(Polygon([(W, S), (E, S), (E, N), (W, N)]))

# remove the empty geometries
intersecting_nodes = gs_nodes[~intersecting_nodes_series.is_empty]
intersecting_edges = gs_edges[~intersecting_edges_series.is_empty]

# for the indices which have new edges, we copy the geometry of the new line on to the top of the old line
indlist = []
dummmy_nodelist = []
for i in range(gs_edges.shape[0]):
    if intersecting_edges_series.iloc[i].wkt != "LINESTRING EMPTY":
        indlist.append(i)
        print("Nodes: U, V: ", gs_edges.index[i][0], gs_edges.index[i][1])
        print(gs_edges.index[i][0] in (gs_nodes.index.tolist()), gs_edges.index[i][1] in (gs_nodes.index.tolist()))
        print("***********")
        u = gs_nodes.loc[gs_edges.index[i][0]]
        v = gs_nodes.loc[gs_edges.index[i][1]]

        plt.scatter(u.x, u.y, color="black")

        # filling an arbitrary highway values and street counts for dummy nodes
        highway = np.nan
        street_count = 3

        x = intersecting_edges_series.iloc[i].xy[0][0]
        y = intersecting_edges_series.iloc[i].xy[1][0]
        # [1.3223464, 103.8527412, 3, nan, <shapely.geometry.point.Point at 0x134a7eb90>]
        to_insert_node = [y, x, street_count, highway, Point(x, y)]

        rand_ = int(np.random.rand() * 1000000000000)
        gs_nodes.loc[rand_] = to_insert_node

        dummmy_nodelist.append(rand_)

        plt.scatter(x, y, color="orange")

        x = intersecting_edges_series.iloc[i].xy[0][-1]
        y = intersecting_edges_series.iloc[i].xy[1][-1]

        # [1.3223464, 103.8527412, 3, nan, <shapely.geometry.point.Point at 0x134a7eb90>]
        to_insert_node = [y, x, street_count, highway, Point(x, y)]

        rand_ = int(np.random.rand() * 1000000000000)
        gs_nodes.loc[rand_] = to_insert_node

        dummmy_nodelist.append(rand_)

        plt.scatter(x, y, color="red")

        gs_edges["geometry"].iloc[i] = intersecting_edges_series.iloc[i]

intersecting_edges = gs_edges[~intersecting_edges_series.is_empty]
intersecting_nodes = gs_nodes[gs_nodes.index.isin(dummmy_nodelist)]


for ind in range(gs_edges.shape[0]):
    if ind in indlist:
        plt.plot(*gs_edges.iloc[ind].geometry.xy, color="green")
    else:
        plt.plot(*gs_edges.iloc[ind].geometry.xy, color="red")
plt.gca().set_aspect("equal")
plt.show()

# for i in range(gs_edges.shape[0]):
#     plt.plot(
#         ((pd.Series(gs_edges[["geometry"]].iloc[i]))[0].xy[0]),
#         ((pd.Series(gs_edges[["geometry"]].iloc[i]))[0].xy[1]),
#         color="black",
#     )
# plt.plot(*(single_bbox_poly_shapely.exterior.xy))
#
# for i in range(intersecting_edges.shape[0]):
#     plt.plot(
#         ((pd.Series(intersecting_edges[["geometry"]].iloc[i]))[0].xy[0]),
#         ((pd.Series(intersecting_edges[["geometry"]].iloc[i]))[0].xy[1]),
#         # color="green",
#     )
# plt.gca().set_aspect("equal")
# plt.show()


graph_attrs = {"crs": "epsg:4326", "simplified": True}
g_truncated = ox.graph_from_gdfs(intersecting_nodes, intersecting_edges, graph_attrs)
fig, ax = ox.plot.plot_graph(
    g_truncated,
    ax=None,
    figsize=(10, 10),
    bgcolor="white",
    node_color="red",
    node_size=5,
    node_alpha=None,
    node_edgecolor="none",
    node_zorder=1,
    edge_color="black",
    edge_linewidth=0.1,
    edge_alpha=None,
    show=False,
    close=False,
    save=False,
    bbox=None,
)
plt.title("Truncated graph")
plt.gca().set_aspect("equal")
plt.savefig("urbanscales/tryouts/g_truncated_new.png", dpi=600)
plt.show()


g_truncated_old = ox.truncate.truncate_graph_bbox(graph, N, S, E, W, truncate_by_edge=False)
ox.plot_graph(g_truncated_old, save=True, filepath="urbanscales/tryouts/truncated_old.png", dpi=600)
