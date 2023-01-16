import sys
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.ops import split
from itertools import chain
import shapely
import time
import pandas as pd
from shapely.geometry import Point, LineString
import osmnx as ox
from osmnx import utils_graph
from smartprint import smartprint as sprint


from shapely.geometry import Point,LineString
from shapely.ops import nearest_points

N, S, E, W = 1.3235381983186159, 1.319982801681384, \
                           103.85361309942331 , 103.84833190057668,
graph = ox.graph_from_bbox(N, S, E, W, \
                           network_type='drive')

# graph = ox.graph_from_point((N, E), dist=500)

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

old_num_edges = gs_edges.shape[0]
old_num_nodes = gs_nodes.shape[0]

int_list = []
edges_to_delete = []
for bbox_line in bbox_lines:
    for i in range(gs_edges.shape[0]):
        a = (pd.Series(gs_edges[["geometry"]].iloc[i]))

        intersection = a[0].intersection(bbox_line)
        int_list.append(intersection)

        if intersection.wkt != "LINESTRING EMPTY":
            print (intersection)
#             ox.utils_geo.interpolate_points(gs_edges, dist)
#             plt.scatter(*intersection.xy, s=30, color="black")

            # x = np.round(int_list[-1].xy[0][0], 7)
            # y = np.round(int_list[-1].xy[1][0], 7)
            x, y = intersection.xy[0][0], intersection.xy[1][0]
            plt.scatter(x, y, s=30, color="black")

            u = gs_edges.index[i][0]
            v = gs_edges.index[i][1]
            key = gs_edges.index[i][2]

            assert u in list(gs_nodes.index) and v in list(gs_nodes.index)

            # create an id for the new node
            v_dash = 10000000000000 + int( np.random.rand()*100000000)

            bbox = (W,S,E,N)
            poly_ = shapely.geometry.box(*bbox, ccw=True)

            # a small buffer polygon is used to remove precision errors in shapely split
            # buffer idea borrowed from user: atkat12's SO answer
            # https://stackoverflow.com/a/50194810/3896008
            buff = Point(x,y).buffer(0.0001)
            split_parts = split(a[0], buff)
            if len(split_parts) == 2:
                first_seg, last_seg = split_parts
            elif len(split_parts) == 3:
                first_seg, buff_seg_unused, last_seg = split_parts
            else:
                print ("Error in splitting; >3 or <2 splits found ")
                raise Exception


            line_with_interpolated_point = LineString(list(first_seg.coords) + list(Point(x,y).coords) + list(last_seg.coords))
            firstLineString, secondLineString = split(line_with_interpolated_point, Point(x, y))

            # plt.plot(*firstLineString.xy, "green")
            # plt.plot(*secondLineString.xy, "blue")
            # plt.show()

            # now we need to figure out which linestring is inside, which is outside the bbox in question
            # so that we can choose the correct end points of the edge to be split
            first_overlap_len = firstLineString.intersection(poly_).length
            second_overlap_len = secondLineString.intersection(poly_).length
            if first_overlap_len > second_overlap_len:
                inside_linestring = firstLineString
                outside_linestring = secondLineString
            elif first_overlap_len < second_overlap_len:
                inside_linestring = firstLineString
                outside_linestring = secondLineString
            else:
                print ("Both linestrings must not be exactly equal; The probability of that happening "
                       "is close to zero. Exiting execution")
                print ("Inside custom truncate function")
                raise Exception

            # similarly we find which out of u and v are inside and
            # outside the bbox respectively
            if poly_.contains(Point(gs_nodes.loc[v].x, gs_nodes.loc[v].y)) and not poly_.contains(Point(gs_nodes.loc[u].x, gs_nodes.loc[u].y)):
                inside_node = v
                outside_node = u
            elif not poly_.contains(Point(gs_nodes.loc[v].x, gs_nodes.loc[v].y)) and poly_.contains(Point(gs_nodes.loc[u].x, gs_nodes.loc[u].y)):
                inside_node = u
                outside_node = v
            else:
                print (poly_.contains(Point(gs_nodes.loc[v].x, gs_nodes.loc[v].y)), poly_.contains(Point(gs_nodes.loc[u].x, gs_nodes.loc[u].y)) )
                time.sleep(1)
                continue
                # ignore the more complicated cases

            e = gs_edges.iloc[i]
            e_dict = {}
            for col in list(gs_edges.columns):
                e_dict[col] = eval("e." + col)

            # we retain all values in the dummy edges except osmid and the geometry (linestring)
            if isinstance(e_dict["osmid"], list):
                e_dict["osmid"].append(-9999) # Since we wish our new edge to have a unique id
            elif isinstance(e_dict["osmid"], int): # some osmid are list, some are int
                e_dict["osmid"] = [e_dict["osmid"], -9999]
            e_dict["geometry"] = inside_linestring
            gs_edges.loc[inside_node, v_dash, key] = e_dict

            # ## no need to create an edge for the outside; We can just add an extra edge inside the box,
            # we retain all values in the dummy edges except osmid and the geometry (linestring)
            if isinstance(e_dict["osmid"], list):
                e_dict["osmid"].append(9999) # Since we wish our new edge to have a unique id
            elif isinstance(e_dict["osmid"], int): # some osmid are list, some are int
                e_dict["osmid"] = [e_dict["osmid"], 9999]  # We use +9999 for outside and -9999 for inside
            e_dict["geometry"] = outside_linestring
            gs_edges.loc[v_dash, outside_node, key] = e_dict

            n = gs_nodes.iloc[0] # choose any node, say the first one
            n_dict = {}
            for col in list(gs_nodes.columns):
                n_dict[col] = eval("n." + col)

            # we retain all values in the dummy node except u, x, and the geometry
            n_dict["geometry"] = Point(x, y)
            n_dict["x"] = x
            n_dict["y"] = y

            gs_nodes.loc[v_dash] = n_dict

            # delete the original node
            # edges_to_delete.append((u,v,key))



        plt.plot(*(a[0]).xy, color="red")
        plt.plot(*(bbox_line.xy), color="green")


new_num_nodes = gs_nodes.shape[0]

for edge in edges_to_delete:
    gs_edges.drop(edge, inplace=True)

# each new node must induce two new edges if considering inside and outside both, otherwise 1
# assert (new_num_nodes - old_num_nodes) * 1 + old_num_edges - len(edges_to_delete) == gs_edges.shape[0]

plt.gca().set_aspect('equal')
plt.savefig("urbanscales/tryouts/Interpolated.png", dpi=600)
plt.show()

for i in range(old_num_edges, gs_edges.shape[0]):
    plt.plot(((pd.Series(gs_edges[["geometry"]].iloc[-i]))[0].xy[0]),
             ((pd.Series(gs_edges[["geometry"]].iloc[-i]))[0].xy[1]), color="black")
for i in range(1, old_num_edges):
    plt.plot(((pd.Series(gs_edges[["geometry"]].iloc[-i]))[0].xy[0]),
             ((pd.Series(gs_edges[["geometry"]].iloc[-i]))[0].xy[1]))

plt.plot(*poly_.exterior.xy, color="green")
for i in range(1, 5):
    # plt.scatter(gs_nodes.iloc[-i].geometry.xy[0], gs_nodes.iloc[-i].geometry.xy[1], s=20, color="blue")
    print (gs_nodes.iloc[-i].x, gs_nodes.iloc[-i].y)
    plt.scatter(gs_nodes.iloc[-i].x, gs_nodes.iloc[-i].y, s=20, color="blue")

plt.gca().set_aspect('equal')
plt.savefig("urbanscales/tryouts/differently_colored_edges.png", dpi=600)
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
plt.savefig("urbanscales/tryouts/Whole_graph_old.png", dpi=600)
plt.show()

graph_attrs = {'crs': 'epsg:4326', 'simplified': True}
graph2 = ox.graph_from_gdfs(gs_nodes, gs_edges, graph_attrs)
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
plt.savefig("urbanscales/tryouts/Whole_graph_new.png", dpi=600)
plt.show()



g_truncated = ox.truncate.truncate_graph_bbox(graph, N, S, E, W, truncate_by_edge=False)
ox.plot_graph(g_truncated, save=True, filepath="urbanscales/tryouts/trunacted_old.png", dpi=600)


#### FINDING THE STACKOVERFLOW USER WHO HELPED WITH THE SOLUTION


g_truncated = ox.truncate.truncate_graph_bbox(graph2,N, S, E, W, truncate_by_edge=False)
ox.plot_graph(g_truncated, save=True, filepath="urbanscales/tryouts/trunacted_new.png", dpi=600)



