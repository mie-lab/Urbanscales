import warnings
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from osmnx import utils_graph
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import Point
from shapely.geometry import Polygon
import config
import geopandas as gdf
from urbanscales.preprocessing.tile import Tile
from smartprint import smartprint as sprint

def smart_truncate(graph, N, S, E, W, get_subgraph=True, get_features=False, scale=-1):

    nodes_orig = ox.graph_to_gdfs(graph, nodes=True, edges=False)
    edges_orig = ox.graph_to_gdfs(graph, edges=True, nodes=False)

    assert get_subgraph != get_features
    if get_features:
        assert scale != -1
        assert not config.rn_plotting_for_truncated_graphs # the plotting is only for debugging

    gs_nodes, gs_edges = utils_graph.graph_to_gdfs(graph)

    bbox_poly = Polygon([(W, S), (E, S), (E, N), (W, N)])
    intersecting_edges_series = gs_edges.intersection(bbox_poly)
    intersecting_edges_series_filtered = intersecting_edges_series[~intersecting_edges_series.geometry.is_empty]

    # if an edge is found in the intersection, it can have only two possibilities
    # 1. It is completely inside (Both u and v are inside the bbox)
    # 2. Either of u or v is inside and the other one is outside
    # Both u and v cannot be outside because if that is the case, the edge must be already
    # split into two parts, and when we retain only the largest connected component, the corner case
    # will be ignored



    # pandas filtering preserves order
    # Also, we make new gdf so that we don't alter the original edges dataframe
    gs_edges_intersecting = gdf.GeoDataFrame(gs_edges[~intersecting_edges_series.geometry.is_empty])

    nodes_to_retain = []

    # filling an arbitrary highway values and street counts for dummy nodes
    highway_dummy = np.nan
    street_count_dummy = 3

    for i in range(intersecting_edges_series_filtered.shape[0]):

    # i = 1
        # Step-1: Get the edge properties from the original edge dataframe
        gs_edges_intersecting["geometry"].iloc[i] = intersecting_edges_series_filtered.iloc[i]
        u,v,key = gs_edges_intersecting.index[i]

        u_x, u_y, v_x, v_y = gs_nodes.loc[u].x, gs_nodes.loc[u].y, gs_nodes.loc[v].x, gs_nodes.loc[v].y

        # Case- I: Both u and v inside the bbox
        if Point(u_x, u_y).within(bbox_poly) and Point(v_x, v_y).within(bbox_poly):
            nodes_to_retain.append(u)
            nodes_to_retain.append(v)
            print ("Both inside")
            # no need to update the u and v for this edge

        elif Point(u_x, u_y).within(bbox_poly) and not Point(v_x, v_y).within(bbox_poly):
            print("U inside; V outside")
            nodes_to_retain.append(u)
            linestring_x, linestring_y = intersecting_edges_series_filtered.iloc[i].xy
            first_point = Point(linestring_x[0], linestring_y[0])
            last_point = Point(linestring_x[-1], linestring_y[-1])
            if first_point.almost_equals(Point(u_x, u_y)) and not last_point.almost_equals(Point(u_x, u_y)) :
                outside_point = last_point
            elif not first_point.almost_equals(Point(u_x, u_y)) and last_point.almost_equals(Point(u_x, u_y)) :
                outside_point = first_point
            else:
                raise Exception("Not implemented; Wrong case found")


            # we must create a dummy node for outside point
            id = int(np.random.rand() * -100000000000000)

            # Format of each node row:
            # [1.3223464, 103.8527412, 3, nan, <shapely.geometry.point.Point at 0x134a7eb90>]
            new_node_data = [outside_point.y, outside_point.y, street_count_dummy, highway_dummy, outside_point]
            gs_nodes.loc[id] = new_node_data

            nodes_to_retain.append(id)

            # Now, we know u is inside and v is outside, (v is replaced by our dummy node)
            # Updating the index in pandas is not possible,
            # So, we insert a copy of the edge at u,v,key into the dataframe
            # And then delete the old edge
            gs_edges_intersecting.loc[(u, id, key)] = gs_edges_intersecting.loc[(u, v, key)]
            gs_edges_intersecting.drop((u, v, key), inplace=True)


        elif not Point(u_x, u_y).within(bbox_poly) and Point(v_x, v_y).within(bbox_poly):
            print("U outside; V inside")
            nodes_to_retain.append(v)
            linestring_x, linestring_y = intersecting_edges_series_filtered.iloc[i].xy
            first_point = Point(linestring_x[0], linestring_y[0])
            last_point = Point(linestring_x[-1], linestring_y[-1])
            if first_point.almost_equals(Point(v_x, v_y)) and not last_point.almost_equals(Point(v_x, v_y)):
                outside_point = last_point
            elif not first_point.almost_equals(Point(v_x, v_y)) and last_point.almost_equals(Point(v_x, v_y)):
                outside_point = first_point
            else:
                raise Exception("Not implemented; Wrong case found")

            # we must create a dummy node for outside point
            id = int(np.random.rand() * -100000000000000)

            # Format of each node row:
            # [1.3223464, 103.8527412, 3, nan, <shapely.geometry.point.Point at 0x134a7eb90>]
            new_node_data = [outside_point.y, outside_point.y, street_count_dummy, highway_dummy, outside_point]
            gs_nodes.loc[id] = new_node_data

            nodes_to_retain.append(id)

            # Now, we know u is outside and v is inside, (u is replaced by our dummy node)
            # Updating the index in pandas is not possible,
            # So, we insert a copy of the edge at u,v,key into the dataframe
            # And then delete the old edge
            gs_edges_intersecting.loc[(id, v, key)] = gs_edges_intersecting.loc[(u, v, key)]
            gs_edges_intersecting.drop((u, v, key), inplace=True)

        elif not Point(u_x, u_y).within(bbox_poly) and not Point(v_x, v_y).within(bbox_poly):
            print ("Both outside, case not handled")
            raise Exception("Both end points of linestring outside the bbox; Case not implemented")



    # for the indices which have new edges, we copy the geometry of the new line on to the top of the old line
    # indlist = []
    # dummmy_nodelist = []
    # for i in range(gs_edges.shape[0]):
    #     if intersecting_edges_series.iloc[i].wkt != "LINESTRING EMPTY":
    #         indlist.append(i)
    #
    #         with warnings.catch_warnings():
    #             # These warnings can be safely ignored
    #             # Refer to user underdark's answer at https://gis.stackexchange.com/a/433423/188758
    #             warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
    #
    #             new_node_edge_start = Point(x, y)
    #             x = intersecting_edges_series.iloc[i].xy[0][0]
    #             y = intersecting_edges_series.iloc[i].xy[1][0]
    #             rand_ = int(np.random.rand() * 1000000000000)
    #             sprint(rand_)
    #             dummmy_nodelist.append(rand_)
    #
    #             # filling an arbitrary highway values and street counts for dummy nodes
    #             highway = np.nan
    #             street_count = 3
    #
    #             # Format of each row:
    #             # [1.3223464, 103.8527412, 3, nan, <shapely.geometry.point.Point at 0x134a7eb90>]
    #             new_node_data = [y, x, street_count, highway, Point(x, y)]
    #             gs_nodes.loc[rand_] = new_node_data
    #
    #
    #             # almost_eq = (gs_nodes.geom_almost_equals(new_node_edge_start))
    #             # if not (almost_eq).any():
    #                 # filling an arbitrary highway values and street counts for dummy nodes
    #                 # highway = np.nan
    #                 # street_count = 3
    #                 #
    #                 # x = intersecting_edges_series.iloc[i].xy[0][0]
    #                 # y = intersecting_edges_series.iloc[i].xy[1][0]
    #                 # rand_ = int(np.random.rand() * 1000000000000)
    #                 # sprint (rand_)
    #                 # dummmy_nodelist.append(rand_)
    #                 # # Format of each row:
    #                 # # [1.3223464, 103.8527412, 3, nan, <shapely.geometry.point.Point at 0x134a7eb90>]
    #                 # to_insert_node = [y, x, street_count, highway, Point(x, y)]
    #                 # gs_nodes.loc[rand_] = to_insert_node
    #                 #
    #
    #
    #                 x = intersecting_edges_series.iloc[i].xy[0][-1]
    #                 y = intersecting_edges_series.iloc[i].xy[1][-1]
    #                 rand_ = int(np.random.rand() * 1000000000000)
    #                 sprint (rand_)
    #                 dummmy_nodelist.append(rand_)
    #                 # Format of each row:
    #                 # [1.3223464, 103.8527412, 3, nan, <shapely.geometry.point.Point at 0x134a7eb90>]
    #                 to_insert_node = [y, x, street_count, highway, Point(x, y)]
    #                 gs_nodes.loc[rand_] = to_insert_node
    #
    #                 # additional step
    #
    #             elif gs_nodes[almost_eq].shape[0] == 1:
    #                 # no updating of nodes (u and v) in this case -> no additional step
    #                 gs_edges["geometry"].iloc[i] = intersecting_edges_series.iloc[i]
    #                 dummmy_nodelist.append(gs_edges.index[i][0])
    #                 dummmy_nodelist.append(gs_edges.index[i][1])
    #
    #
    #
    #
    #         if config.rn_plotting_for_truncated_graphs:
    #             plt.scatter(x, y, color="orange")
    #
    #
    #         if config.rn_plotting_for_truncated_graphs:
    #             plt.scatter(x, y, color="orange")
    #

    # intersecting_edges = gs_edges[~intersecting_edges_series.is_empty]
    intersecting_nodes = gs_nodes[gs_nodes.index.isin(nodes_to_retain)]
    intersecting_edges = gs_edges_intersecting

    graph_attrs = {"crs": "epsg:4326", "simplified": True}
    g_truncated = ox.graph_from_gdfs(intersecting_nodes, intersecting_edges, graph_attrs)
    if config.rn_plotting_for_truncated_graphs:
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

        orig_graph = ox.graph_from_gdfs(nodes_orig, edges_orig, graph_attrs)

        g_truncated_old = ox.truncate.truncate_graph_bbox(orig_graph, N, S, E, W, truncate_by_edge=False)
        fig, ax = ox.plot_graph(g_truncated_old, save=True, filepath="urbanscales/tryouts/truncated_old.png", dpi=600)
        plt.title("Truncated graph old")
        plt.gca().set_aspect("equal")
        plt.savefig("urbanscales/tryouts/g_truncated_old.png", dpi=600)
        plt.show()
    #
    #     return ox.utils_graph.get_largest_component(g_truncated), ox.utils_graph.get_largest_component(g_truncated_old)
    #
    if get_features:
        t = Tile((g_truncated), \
                 (config.rn_square_from_city_centre ** 2) / (scale ** 2))
        return t.get_vector_of_features()
    elif get_subgraph:
        return ox.utils_graph.get_largest_component(g_truncated)

if __name__ =="__main__":
    N, S, E, W = (
        1.3235381983186159,
        1.319982801681384,
        103.85361309942331,
        103.84833190057668,
    )

    graph = ox.graph_from_bbox(N, S, E, W, network_type="drive")
    sprint (N, S, E, W)

    h = N-S
    w = E - W
    S = S + h*0.5
    W = W + w*0.5
    N = S + h * 0.4
    E = W + w*0.4
    sprint (N, S, E, W)

    graph = smart_truncate(graph, N, S, E, W, get_features=False, get_subgraph=True, scale=25, )

    debug_pitstop = True


