import os.path

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from osmnx import utils_graph
from shapely.errors import ShapelyDeprecationWarning
from tqdm import tqdm

import warnings


warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
from shapely.geometry import Point
from shapely.geometry import Polygon
import config
import geopandas as gdf
from urbanscales.preprocessing.tile import Tile
from smartprint import smartprint as sprint
import time
from line_profiler_pycharm import profile

global_dummies = list(set((np.random.rand(100000) * 100000000000).astype("int").tolist()))
# list.sort(global_dummies)


@profile
def smart_truncate(graph, gs_nodes, gs_edges, N, S, E, W, get_subgraph=True, get_features=False, scale=-1):
    gs_nodes, gs_edges = gdf.GeoDataFrame(gs_nodes), gdf.GeoDataFrame(gs_edges)

    starttime = time.time()

    if config.rn_plotting_for_truncated_graphs:
        plt.gca().set_aspect("equal")
        upto_9_colors = "bgrcmykw"

        nodes_orig = ox.graph_to_gdfs(graph, nodes=True, edges=False)
        edges_orig = ox.graph_to_gdfs(graph, edges=True, nodes=False)

    assert get_subgraph != get_features
    if get_features:
        assert scale != -1
        assert not config.rn_plotting_for_truncated_graphs  # the plotting is only for debugging

    try:
        count_dummies = smart_truncate.count_dummies
        smart_truncate.count_dummies += 1
    except AttributeError:
        count_dummies = 0
        smart_truncate.count_dummies = 0

    # try:
    #     gs_nodes, gs_edges = smart_truncate.gs_nodes_gs_edges
    #     # print("Try pass")
    # except AttributeError:
    #     smart_truncate.gs_nodes_gs_edges = utils_graph.graph_to_gdfs(graph)
    #     gs_nodes, gs_edges = smart_truncate.gs_nodes_gs_edges
    # print("Try fail, values recomputed")

    # global_dummies = list(range(gs_nodes.index.max()+1, gs_nodes.index.max() + 10000))
    # for insert_placeholders in tqdm(range(100), desc="creating placeholders"):
    #     # sprint (global_dummies[insert_placeholders])
    #     gs_nodes.loc[global_dummies[insert_placeholders]] = 0 # [1, 1, 1, 1, 1, 1]

    # for insert_placeholders in tqdm(range(100), desc="creating placeholders known indices"):
    #     # sprint (global_dummies[insert_placeholders])
    #     gs_nodes.loc[gs_nodes.index[0]] = 0 # [1, 1, 1, 1, 1, 1]

    to_add_in_gs_nodes = []
    to_add_in_gs_nodes_indices = []
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

    edge_indices_to_drop = []

    # for i in tqdm(range(intersecting_edges_series_filtered.shape[0]), desc="edges iterate: "):
    for i in range(intersecting_edges_series_filtered.shape[0]):
        # Step-1: Get the edge properties from the original edge dataframe
        gs_edges_intersecting["geometry"].iloc[i] = intersecting_edges_series_filtered.iloc[i]
        u, v, key = gs_edges_intersecting.index[i]

        if config.DEBUG_TRUNCATE:
            sprint(gs_edges_intersecting.index)
            sprint(u, v, key, i)

        if config.rn_plotting_for_truncated_graphs:
            plt.plot(*gs_edges.loc[u, v, key].geometry.xy, color=upto_9_colors[i])
            plt.plot(*intersecting_edges_series_filtered.iloc[i].xy, color=upto_9_colors[i], linewidth=5, alpha=0.4)

        u_x, u_y, v_x, v_y = gs_nodes.loc[u].x, gs_nodes.loc[u].y, gs_nodes.loc[v].x, gs_nodes.loc[v].y

        # Case- I: Both u and v inside the bbox
        if Point(u_x, u_y).within(bbox_poly) and Point(v_x, v_y).within(bbox_poly):
            nodes_to_retain.append(u)
            nodes_to_retain.append(v)
            if config.DEBUG_TRUNCATE:
                print("Both inside")
            # no need to update the u and v for this edge

        elif Point(u_x, u_y).within(bbox_poly) and not Point(v_x, v_y).within(bbox_poly):
            if config.DEBUG_TRUNCATE:
                print("U inside; V outside")
            nodes_to_retain.append(u)
            linestring_x, linestring_y = intersecting_edges_series_filtered.iloc[i].xy
            first_point = Point(linestring_x[0], linestring_y[0])
            last_point = Point(linestring_x[-1], linestring_y[-1])
            if first_point.within(bbox_poly) and not last_point.within(bbox_poly):
                outside_point = last_point
            elif not first_point.within(bbox_poly) and last_point.within(bbox_poly):
                outside_point = first_point
            else:
                raise Exception("Not implemented; Wrong case found")

            # we must create a dummy node for outside point
            id = global_dummies[count_dummies]  # int(np.random.rand() * -100000000000000)
            count_dummies += 1

            # Format of each node row:
            # [1.3223464, 103.8527412, 3, nan, <shapely.geometry.point.Point at 0x134a7eb90>]
            new_node_data = [outside_point.y, outside_point.x, street_count_dummy, highway_dummy, outside_point]
            if "ref" in gs_nodes.columns:
                new_node_data[4:4] = [np.nan]
            # gs_nodes.loc[id] = new_node_data
            to_add_in_gs_nodes.append(new_node_data)
            to_add_in_gs_nodes_indices.append(id)

            nodes_to_retain.append(id)

            # Now, we know u is inside and v is outside, (v is replaced by our dummy node)
            # Updating the index in pandas is not possible,
            # So, we insert a copy of the edge at u,v,key into the dataframe
            # And then delete the old edge
            gs_edges_intersecting.loc[(u, id, key)] = gs_edges_intersecting.loc[(u, v, key)]
            # gs_edges_intersecting.drop((u, v, key), inplace=True)
            edge_indices_to_drop.append((u, v, key))

        elif not Point(u_x, u_y).within(bbox_poly) and Point(v_x, v_y).within(bbox_poly):
            if config.DEBUG_TRUNCATE:
                print("U outside; V inside")
            nodes_to_retain.append(v)
            linestring_x, linestring_y = intersecting_edges_series_filtered.iloc[i].xy
            first_point = Point(linestring_x[0], linestring_y[0])
            last_point = Point(linestring_x[-1], linestring_y[-1])
            if first_point.within(bbox_poly) and not last_point.within(bbox_poly):
                outside_point = last_point
            elif not first_point.within(bbox_poly) and last_point.within(bbox_poly):
                outside_point = first_point
            else:
                raise Exception("Not implemented; Wrong case found")

            # we must create a dummy node for outside point
            id = global_dummies[count_dummies]  # int(np.random.rand() * -100000000000000)
            count_dummies += 1

            # Format of each node row:
            # [1.3223464, 103.8527412, 3, nan, <shapely.geometry.point.Point at 0x134a7eb90>]
            new_node_data = [outside_point.y, outside_point.x, street_count_dummy, highway_dummy, outside_point]
            if "ref" in gs_nodes.columns:
                new_node_data[4:4] = [np.nan]
            # gs_nodes.loc[id] = new_node_data
            to_add_in_gs_nodes.append(new_node_data)
            to_add_in_gs_nodes_indices.append(id)

            # for ki in tqdm(range(100), desc="loc insert"):
            #     gs_nodes.loc[id + ki] = new_node_data

            nodes_to_retain.append(id)

            # Now, we know v is inside and u is outside, (u is replaced by our dummy node)
            # Updating the index in pandas is not possible,
            # So, we insert a copy of the edge at u,v,key into the dataframe
            # And then delete the old edge
            gs_edges_intersecting.loc[(id, v, key)] = gs_edges_intersecting.loc[(u, v, key)]
            if config.DEBUG_TRUNCATE:
                sprint((id, v, key))
            edge_indices_to_drop.append((u, v, key))

        elif not Point(u_x, u_y).within(bbox_poly) and not Point(v_x, v_y).within(bbox_poly):
            if config.DEBUG_TRUNCATE:
                print("U inside; V outside")
            # no nodes to retain
            # nodes_to_retain.append(u)

            linestring_x, linestring_y = intersecting_edges_series_filtered.iloc[i].xy
            first_point = Point(linestring_x[0], linestring_y[0])
            last_point = Point(linestring_x[-1], linestring_y[-1])

            # For this case, the edge passes through the bbox
            # The intersection points should touch the bbox
            try:
                assert (first_point.touches(bbox_poly)) and (last_point.touches(bbox_poly))
            except:
                raise Exception("Passing through egde, error")

            # we must create a dummy node for outside point
            id1 = global_dummies[count_dummies]  # int(np.random.rand() * -100000000000000)
            count_dummies += 1

            # Format of each node row:
            # [1.3223464, 103.8527412, 3, nan, <shapely.geometry.point.Point at 0x134a7eb90>]
            outside_point = first_point
            new_node_data = [outside_point.y, outside_point.x, street_count_dummy, highway_dummy, outside_point]
            if "ref" in gs_nodes.columns:
                new_node_data[4:4] = [np.nan]
            # gs_nodes.loc[id1] = new_node_data
            to_add_in_gs_nodes.append(new_node_data)
            to_add_in_gs_nodes_indices.append(id1)

            nodes_to_retain.append(id1)

            id2 = global_dummies[count_dummies]  # int(np.random.rand() * -100000000000000)
            count_dummies += 1

            outside_point = last_point
            new_node_data = [outside_point.y, outside_point.x, street_count_dummy, highway_dummy, outside_point]
            if "ref" in gs_nodes.columns:
                new_node_data[4:4] = [np.nan]
            # gs_nodes.loc[id2] = new_node_data
            to_add_in_gs_nodes.append(new_node_data)
            to_add_in_gs_nodes_indices.append(id2)

            nodes_to_retain.append(id2)

            # Now, we know u is inside and v is outside, (v is replaced by our dummy node)
            # Updating the index in pandas is not possible,
            # So, we insert a copy of the edge at u,v,key into the dataframe
            # And then delete the old edge
            gs_edges_intersecting.loc[(id1, id2, key)] = gs_edges_intersecting.loc[(u, v, key)]
            # gs_edges_intersecting.drop((u, v, key), inplace=True)
            edge_indices_to_drop.append((u, v, key))

    # if config.rn_plotting_for_truncated_graphs:
    #     plt.savefig(os.path.join(config.BASE_FOLDER, config.results_folder, "_debug_trunc" + str(int(np.random.rand() *1000000)) + ".png"))
    #     # plt.show()
    #     plt.clf()

    gs_nodes = gs_nodes.reindex(index=gs_nodes.index.append(pd.Index(to_add_in_gs_nodes_indices)))
    for index, dummy_node_data in enumerate(to_add_in_gs_nodes_indices):
        gs_nodes.loc[index] = dummy_node_data

    intersecting_nodes = gs_nodes[gs_nodes.index.isin((set(nodes_to_retain)))]
    intersecting_edges = gs_edges_intersecting[~gs_edges_intersecting.index.isin((set(edge_indices_to_drop)))]

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

    print("Time taken in smart_truncate_gpd call: ", (time.time() - starttime), " seconds")
    smart_truncate.count_dummies = count_dummies
    if get_features:
        t = Tile((g_truncated), (config.rn_square_from_city_centre ** 2) / (scale ** 2))
        return t.get_vector_of_features()
    elif get_subgraph:
        return ox.utils_graph.get_largest_component(g_truncated)


if __name__ == "__main__":
    # N, S, E, W = (
    #     1.3235381983186159,
    #     1.319982801681384,
    #     103.85361309942331,
    #     103.84833190057668,
    # )

    # graph = ox.graph_from_bbox(N, S, E, W, network_type="drive")
    # sprint(N, S, E, W)
    #
    # h = N - S
    # w = E - W
    # S = S + h * 0.5
    # W = W + w * 0.5
    # N = S + h * 0.4
    # E = W + w * 0.4
    # sprint(N, S, E, W)

    # for multiple_calls in range(100):
    #     graph = smart_truncate(
    #         graph,
    #         N,
    #         S,
    #         E,
    #         W,
    #         get_features=False,
    #         get_subgraph=True,
    #         scale=25,
    #     )

    # The imports moved here to avoid circular import
    from urbanscales.io.road_network import RoadNetwork

    rn = RoadNetwork("Singapore")
    graph = rn.G_osm
    N, S, E, W = rn.max_y, rn.min_y, rn.max_x, rn.min_x
    h = N - S
    w = E - W
    S = S + h * 0.1
    W = W + w * 0.1
    N = S + h * 0.1
    E = W + w * 0.1
    sprint(N, S, E, W)

    for i in range(2):
        truncated_graph = smart_truncate(
            graph,
            N,
            S,
            E,
            W,
            get_features=False,
            get_subgraph=True,
            scale=25,
        )

    debug_pitstop = True
