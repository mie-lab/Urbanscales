
import numpy as np
import osmnx as ox
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt

N, S, E, W = 1.3235381983186159, 1.319982801681384, \
                           103.85361309942331 , 103.84833190057668,
graph = ox.graph_from_bbox(N, S, E, W, \
                           network_type='drive')
nodes= ox.graph_to_gdfs(graph, nodes=True, edges=False)
edges= ox.graph_to_gdfs(graph, edges=True, nodes=False)
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
W_ = W + (E-W) * 0.8
S_ = S + (N-S)*0.7
width = (E - W)*0.07 
height = (N - S)*0.1 

rect = plt.Rectangle((W_, S_), width, height, facecolor="green", alpha=0.3, edgecolor=None)
ax.add_patch(rect)
plt.show()

g_truncated = ox.truncate.truncate_graph_bbox(graph, S_ + height, S_, W_+width, W_, truncate_by_edge=False)
ox.plot_graph(g_truncated)




import pandas as pd
from shapely.geometry import Point, LineString
import osmnx as ox 
from osmnx import utils_graph
from smartprint import smartprint as sprint 


G = graph
gs_nodes, gs_edges = utils_graph.graph_to_gdfs(G)
sprint (gs_nodes.shape)

E = W_+width 
W = W_ 
S = S_ 
N = S_ + height
bbox_lines = [
              LineString([Point(W,S), Point(E,S)]), \
              LineString([Point(E,S), Point(E,N)]), \
              LineString([Point(E,N), Point(W,N)]), \
              LineString([Point(W,N), Point(W,S)]), \
             ]

int_list = []
for bbox_line in bbox_lines:
    for i in range(gs_edges.shape[0]):
        a = (pd.Series(gs_edges[["geometry"]].iloc[i]))

        intersection = a[0].intersection(bbox_line)
        int_list.append(intersection)
        if intersection.wkt != "LINESTRING EMPTY":
            print (intersection)
#             ox.utils_geo.interpolate_points(gs_edges, dist)
            plt.scatter(*intersection.xy, s=30, color="black")

            x = np.round(int_list[-1].xy[0][0], 6)
            y = np.round(int_list[-1].xy[1][0], 6)
            
            new_row = [y, x, 3, np.nan, Point(x,y)]

            gs_nodes = gs_nodes.append(pd.Series(new_row, index=gs_nodes.columns\
                                      [:len(new_row)]), ignore_index=True)
            
        plt.plot(*(a[0]).xy, color="red")
        plt.plot(*(bbox_line.xy), color="green")
plt.gca().set_aspect('equal')
plt.savefig("Interpolated.png", dpi=600)
plt.show()


fig, ax = ox.plot.plot_graph(
                G,
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
plt.title("Whole graph before introducing new nodes")
plt.gca().set_aspect('equal')

# ox.utils_geo.round_geometry_coords(gs_nodes, precision=6)
graph2 = ox.graph_from_gdfs(gs_nodes, gs_edges)
fig, ax = ox.plot.plot_graph(
                graph2,
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
plt.title("Whole graph after introducing new nodes")
plt.gca().set_aspect('equal')


