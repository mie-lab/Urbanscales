import os.path
import pickle
import sys
import time
import shapely
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from osmnx import utils_graph
from shapely.errors import ShapelyDeprecationWarning
import warnings
from urbanscales.io.road_network import RoadNetwork
import logging
import networkx

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

from shapely.geometry import Point
from shapely.geometry import Polygon
import config

logging.basicConfig(filename=config.log_file)

import geopandas as gdf
from urbanscales.preprocessing.tile import Tile
from smartprint import smartprint as sprint

# from line_profiler_pycharm import profile


# @profile
def smart_truncate(
    graph, gs_nodes, gs_edges, N, S, E, W, get_subgraph=True, get_features=False, scale=-1, legacy=False
):
    ss = time.time()
    """

    Args:
        graph:
        gs_nodes:
        gs_edges:
        N:
        S:
        E:
        W:
        get_subgraph:
        get_features:
        scale:
        legacy: If True, revertes back to the more inefficient code; which is easier to debug

    Returns:

    """
    gs_nodes, gs_edges = gdf.GeoDataFrame(gs_nodes), gdf.GeoDataFrame(gs_edges)
    # graph = gdf.GeoDataFrame(graph)

    if config.rn_plotting_for_truncated_graphs:
        plt.gca().set_aspect("equal")

    if config.rn_plotting_for_truncated_graphs:
        nodes_orig = gs_nodes
        edges_orig = gs_edges

    assert get_subgraph != get_features
    if get_features:
        assert scale != -1
        assert not config.rn_plotting_for_truncated_graphs  # the plotting is only for debugging

    bbox_poly = Polygon([(W, S), (E, S), (E, N), (W, N)])
    h = N - S
    w = E - W

    # reduce the search space; Add 25% on each side (N-S and E-W)
    if not legacy:
        gs_edges = gs_edges.cx[W - h / 4 : E + h / 4, S - w / 4 : N + w / 4]

    intersecting_edges_series = gs_edges.intersection(bbox_poly)

    # intersecting_edges_series = intersecting_edges_series_1
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

    dict_gs_nodes_insert = {}
    dict_gs_edges_intersecting = {}

    for i in range(intersecting_edges_series_filtered.shape[0]):
        # Step-1: Get the edge properties from the original edge dataframe
        gs_edges_intersecting["geometry"].iloc[i] = intersecting_edges_series_filtered.iloc[i]
        u, v, key = gs_edges_intersecting.index[i]

        if config.DEBUG:
            sprint(gs_edges_intersecting.index)
            sprint(u, v, key, i)

        if isinstance(intersecting_edges_series_filtered.iloc[i], shapely.geometry.LineString) or isinstance(
            intersecting_edges_series_filtered.iloc[i], shapely.geometry.Point
        ):
            if config.rn_plotting_for_truncated_graphs:
                plt.plot(*gs_edges.loc[u, v, key].geometry.xy)  # , color=upto_9_colors[i])
                plt.plot(
                    *intersecting_edges_series_filtered.iloc[i].xy, linewidth=5, alpha=0.4
                )  # color=upto_9_colors[i],

        u_x, u_y, v_x, v_y = gs_nodes.loc[u].x, gs_nodes.loc[u].y, gs_nodes.loc[v].x, gs_nodes.loc[v].y

        # Case- I: Both u and v inside the bbox
        if Point(u_x, u_y).within(bbox_poly) and Point(v_x, v_y).within(bbox_poly):
            nodes_to_retain.append(u)
            nodes_to_retain.append(v)
            if config.DEBUG:
                print("Both inside")
            # no need to update the u and v for this edge

        elif Point(u_x, u_y).within(bbox_poly) and not Point(v_x, v_y).within(bbox_poly):
            if config.DEBUG:
                print("U inside; V outside")
            nodes_to_retain.append(u)
            if isinstance(intersecting_edges_series_filtered.iloc[i], shapely.geometry.LineString) or isinstance(
                intersecting_edges_series_filtered.iloc[i], shapely.geometry.Point
            ):
                linestring_x, linestring_y = intersecting_edges_series_filtered.iloc[i].xy

                if config.LOGGING_ENABLED:
                    with open(config.log_file, "a") as f:
                        f.write("Standard case\n")

            elif isinstance(intersecting_edges_series_filtered.iloc[i], shapely.geometry.MultiLineString):
                xy_linestring = (
                    intersecting_edges_series_filtered.iloc[i]
                    .wkt.replace("MULTIL", "L")
                    .replace("),", ",")
                    .replace(", (", ",")
                    .replace("((", "(")
                    .replace("))", ")")
                )
                if config.LOGGING_ENABLED:
                    with open(config.log_file, "a") as f:
                        f.write("..................Multiline Case\n")

                linestring_x, linestring_y = shapely.wkt.loads(xy_linestring).xy

            first_point = Point(linestring_x[0], linestring_y[0])
            last_point = Point(linestring_x[-1], linestring_y[-1])
            if first_point.within(bbox_poly) and not last_point.within(bbox_poly):
                outside_point = last_point
            elif not first_point.within(bbox_poly) and last_point.within(bbox_poly):
                outside_point = first_point
            else:
                raise Exception("Not implemented; Wrong case found")

            # we must create a dummy node for outside point
            id = int(np.random.rand() * -100000000000000)

            # Format of each node row:
            # [1.3223464, 103.8527412, 3, nan, <shapely.geometry.point.Point at 0x134a7eb90>]
            new_node_data = [outside_point.y, outside_point.x, street_count_dummy, highway_dummy, outside_point]
            if "ref" in gs_nodes.columns:
                new_node_data[4:4] = [np.nan]

            if legacy:
                gs_nodes.loc[id] = new_node_data

            dict_gs_nodes_insert[id] = new_node_data

            nodes_to_retain.append(id)

            # Now, we know u is inside and v is outside, (v is replaced by our dummy node)
            # Updating the index in pandas is not possible,
            # So, we insert a copy of the edge at u,v,key into the dataframe
            # And then delete the old edge

            if legacy:
                gs_edges_intersecting.loc[(u, id, key)] = gs_edges_intersecting.loc[(u, v, key)]

            dict_gs_edges_intersecting[(u, id, key)] = gs_edges_intersecting.loc[(u, v, key)]

            edge_indices_to_drop.append((u, v, key))

        elif not Point(u_x, u_y).within(bbox_poly) and Point(v_x, v_y).within(bbox_poly):
            if config.DEBUG:
                print("U outside; V inside")
            nodes_to_retain.append(v)

            if isinstance(intersecting_edges_series_filtered.iloc[i], shapely.geometry.LineString) or isinstance(
                intersecting_edges_series_filtered.iloc[i], shapely.geometry.Point
            ):
                linestring_x, linestring_y = intersecting_edges_series_filtered.iloc[i].xy

                if config.LOGGING_ENABLED:
                    with open(config.log_file, "a") as f:
                        f.write("Standard case\n")

            elif isinstance(intersecting_edges_series_filtered.iloc[i], shapely.geometry.MultiLineString):
                # convert multi line string to linestring
                xy_linestring = (
                    intersecting_edges_series_filtered.iloc[i]
                    .wkt.replace("MULTIL", "L")
                    .replace("),", ",")
                    .replace(", (", ",")
                    .replace("((", "(")
                    .replace("))", ")")
                )
                if config.LOGGING_ENABLED:
                    with open(config.log_file, "a") as f:
                        f.write("..................Multiline Case\n")

                linestring_x, linestring_y = shapely.wkt.loads(xy_linestring).xy

            else:
                raise Exception("Some other type of geometry present")

            first_point = Point(linestring_x[0], linestring_y[0])
            last_point = Point(linestring_x[-1], linestring_y[-1])
            if first_point.within(bbox_poly) and not last_point.within(bbox_poly):
                outside_point = last_point
            elif not first_point.within(bbox_poly) and last_point.within(bbox_poly):
                outside_point = first_point
            else:
                raise Exception("Not implemented; Wrong case found")

            # we must create a dummy node for outside point
            id = int(np.random.rand() * -100000000000000)

            # Format of each node row:
            # [1.3223464, 103.8527412, 3, nan, <shapely.geometry.point.Point at 0x134a7eb90>]
            new_node_data = [outside_point.y, outside_point.x, street_count_dummy, highway_dummy, outside_point]
            if "ref" in gs_nodes.columns:
                new_node_data[4:4] = [np.nan]

            if legacy:
                gs_nodes.loc[id] = new_node_data

            dict_gs_nodes_insert[id] = new_node_data

            nodes_to_retain.append(id)

            # Now, we know v is inside and u is outside, (u is replaced by our dummy node)
            # Updating the index in pandas is not possible,
            # So, we insert a copy of the edge at u,v,key into the dataframe
            # And then delete the old edge

            if legacy:
                gs_edges_intersecting.loc[(id, v, key)] = gs_edges_intersecting.loc[(u, v, key)]

            dict_gs_edges_intersecting[(id, v, key)] = gs_edges_intersecting.loc[(u, v, key)]

            if config.DEBUG:
                sprint((id, v, key))
            edge_indices_to_drop.append((u, v, key))

        elif not Point(u_x, u_y).within(bbox_poly) and not Point(v_x, v_y).within(bbox_poly):
            if config.DEBUG:
                print("U outside; V outside")
            # no nodes to retain

            if isinstance(intersecting_edges_series_filtered.iloc[i], shapely.geometry.LineString) or isinstance(
                intersecting_edges_series_filtered.iloc[i], shapely.geometry.Point
            ):
                linestring_x, linestring_y = intersecting_edges_series_filtered.iloc[i].xy

                if config.LOGGING_ENABLED:
                    with open(config.log_file, "a") as f:
                        f.write("Standard Case\n")

            elif isinstance(intersecting_edges_series_filtered.iloc[i], shapely.geometry.MultiLineString):
                # convert multi line string to linestring
                xy_linestring = (
                    intersecting_edges_series_filtered.iloc[i]
                    .wkt.replace("MULTIL", "L")
                    .replace("),", ",")
                    .replace(", (", ",")
                    .replace("((", "(")
                    .replace("))", ")")
                )

                if config.LOGGING_ENABLED:
                    with open(config.log_file, "a") as f:
                        f.write("..................Multiline Case\n")

                linestring_x, linestring_y = shapely.wkt.loads(xy_linestring).xy

            else:
                raise Exception("Some other type of geometry present")

            first_point = Point(linestring_x[0], linestring_y[0])
            last_point = Point(linestring_x[-1], linestring_y[-1])

            # For this case, the edge passes through the bbox
            # The intersection points should touch the bbox
            try:
                assert (first_point.touches(bbox_poly)) and (last_point.touches(bbox_poly))
            except:
                raise Exception("Passing through egde, error")

            # we must create a dummy node for outside point
            id1 = int(np.random.rand() * -100000000000000)

            # Format of each node row:
            # [1.3223464, 103.8527412, 3, nan, <shapely.geometry.point.Point at 0x134a7eb90>]
            outside_point = first_point
            new_node_data = [outside_point.y, outside_point.x, street_count_dummy, highway_dummy, outside_point]
            if "ref" in gs_nodes.columns:
                new_node_data[4:4] = [np.nan]

            if legacy:
                gs_nodes.loc[id1] = new_node_data

            dict_gs_nodes_insert[id1] = new_node_data

            nodes_to_retain.append(id1)

            id2 = int(np.random.rand() * -100000000000000)
            outside_point = last_point
            new_node_data = [outside_point.y, outside_point.x, street_count_dummy, highway_dummy, outside_point]
            if "ref" in gs_nodes.columns:
                new_node_data[4:4] = [np.nan]

            if legacy:
                gs_nodes.loc[id2] = new_node_data

            dict_gs_nodes_insert[id2] = new_node_data

            nodes_to_retain.append(id2)

            # Now, we know u is inside and v is outside, (v is replaced by our dummy node)
            # Updating the index in pandas is not possible,
            # So, we insert a copy of the edge at u,v,key into the dataframe
            # And then delete the old edge

            if legacy:
                gs_edges_intersecting.loc[(id1, id2, key)] = gs_edges_intersecting.loc[(u, v, key)]

            dict_gs_edges_intersecting[(id1, id2, key)] = gs_edges_intersecting.loc[(u, v, key)]

            edge_indices_to_drop.append((u, v, key))

    if not legacy:
        # bulk insert 1
        if len(dict_gs_nodes_insert) > 0:
            df_dictionary = pd.DataFrame(dict_gs_nodes_insert).T
            df_dictionary.columns = gs_nodes.columns
            gs_nodes = pd.concat([gs_nodes, df_dictionary], join="outer")

        # bulk insert 2
        if len(dict_gs_edges_intersecting) > 0:
            df_dictionary = pd.DataFrame(dict_gs_edges_intersecting).T
            df_dictionary.columns = gs_edges_intersecting.columns
            gs_edges_intersecting = pd.concat([gs_edges_intersecting, df_dictionary], join="outer")

    # if config.rn_plotting_for_truncated_graphs:
    #     plt.show(block=False)

    intersecting_nodes = gs_nodes[gs_nodes.index.isin(list(set(nodes_to_retain)))]
    intersecting_edges = gs_edges_intersecting[~gs_edges_intersecting.index.isin(list(set(edge_indices_to_drop)))]

    graph_attrs = {"crs": "epsg:4326", "simplified": True}
    g_truncated = ox.graph_from_gdfs(intersecting_nodes, intersecting_edges, graph_attrs)
    if config.rn_plotting_for_truncated_graphs:
        try:
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
        except ValueError:
            return config.rn_no_stats_marker
        plt.title("Truncated graph")
        plt.gca().set_aspect("equal")
        plot_num = int(np.random.rand() * 100000000000000)
        plt.savefig(
            os.path.join(
                config.BASE_FOLDER,
                "urbanscales/tryouts/smart_truncated_plots/g_truncated_new_" + str(plot_num) + ".png",
            ),
            dpi=600,
        )  # plt.show(block=False)

        orig_graph = ox.graph_from_gdfs(nodes_orig, edges_orig, graph_attrs)

        plt.clf()
        try:
            g_truncated_old = ox.truncate.truncate_graph_bbox(orig_graph, N, S, E, W, truncate_by_edge=False)
        except networkx.exception.NetworkXPointlessConcept:
            return config.rn_no_stats_marker
        fig, ax = ox.plot.plot_graph(
            g_truncated_old,
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

        plt.title("Truncated graph old")
        plt.gca().set_aspect("equal")
        plt.savefig(
            os.path.join(
                config.BASE_FOLDER,
                "urbanscales/tryouts/smart_truncated_plots/g_truncated_old_" + str(plot_num) + ".png",
            ),
            dpi=600,
        )
        # plt.show(block=False)

    if networkx.is_empty(g_truncated):
        print("Null graph returned")
        return config.rn_no_stats_marker

    if config.DEBUG:
        print("Inside the function: ", time.time() - ss)

    if get_features:
        t = Tile((g_truncated), (config.rn_square_from_city_centre**2) / (scale**2))
        return t.get_vector_of_features()
    elif get_subgraph:
        return ox.utils_graph.get_largest_component(g_truncated)


if __name__ == "__main__":
    """
    for scale in [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6]:
        for legacy in [True, False]:


            N, S, E, W = (
                1.3695873752225538,
                1.3132978779946807,
                103.84008497795305,
                103.88540358018281,
            )

            graph = ox.graph_from_bbox(N, S, E, W, network_type="drive")
            # sprint(N, S, E, W)

            h = N - S
            w = E - W
            S = S + h * scale
            W = W + w * scale
            N = S + h * scale
            E = W + w * scale
            # sprint(N, S, E, W)

            gs_nodes, gs_edges = utils_graph.graph_to_gdfs(graph)

            starttime = time.time()
            for repeat in range(10):
                graph = smart_truncate(
                    graph, gs_nodes, gs_edges, N, S, E, W, get_features=True, get_subgraph=False, scale=25, legacy=legacy
                )

            debug_pitstop = True

            sprint(scale, legacy,  (time.time() - starttime)/10)
            sprint(scale, legacy,  (time.time() - starttime)/10)
    """
    sprint(os.getcwd())
    os.system("rm urbanscales/tryouts/smart_truncated_plots/*.png")

    if os.path.exists(config.log_file):
        os.remove(config.log_file)
    print("Cleaned the log file")

    list_of_bbox = [
        (1.2897639799999998, 1.2684322, 103.79295100109655, 103.7717563, 35),
        (1.2897639799999998, 1.2684322, 103.8141457021931, 103.79295100109655, 35),
        (1.2897639799999998, 1.2684322, 103.83534040328965, 103.8141457021931, 35),
        (1.2897639799999998, 1.2684322, 103.8565351043862, 103.83534040328965, 35),
        (1.2897639799999998, 1.2684322, 103.87772980548274, 103.8565351043862, 35),
        (1.2897639799999998, 1.2684322, 103.89892450657929, 103.87772980548274, 35),
        (1.2897639799999998, 1.2684322, 103.92011920767584, 103.89892450657929, 35),
        (1.3110957599999997, 1.2897639799999998, 103.79295100109655, 103.7717563, 35),
        (1.3110957599999997, 1.2897639799999998, 103.8141457021931, 103.79295100109655, 35),
        (1.3110957599999997, 1.2897639799999998, 103.83534040328965, 103.8141457021931, 35),
        (1.3110957599999997, 1.2897639799999998, 103.8565351043862, 103.83534040328965, 35),
        (1.3110957599999997, 1.2897639799999998, 103.87772980548274, 103.8565351043862, 35),
        (1.3110957599999997, 1.2897639799999998, 103.89892450657929, 103.87772980548274, 35),
        (1.3110957599999997, 1.2897639799999998, 103.92011920767584, 103.89892450657929, 35),
        (1.3324275399999996, 1.3110957599999997, 103.79295100109655, 103.7717563, 35),
        (1.3324275399999996, 1.3110957599999997, 103.8141457021931, 103.79295100109655, 35),
        (1.3324275399999996, 1.3110957599999997, 103.83534040328965, 103.8141457021931, 35),
        (1.3324275399999996, 1.3110957599999997, 103.8565351043862, 103.83534040328965, 35),
        (1.3324275399999996, 1.3110957599999997, 103.87772980548274, 103.8565351043862, 35),
        (1.3324275399999996, 1.3110957599999997, 103.89892450657929, 103.87772980548274, 35),
        (1.3324275399999996, 1.3110957599999997, 103.92011920767584, 103.89892450657929, 35),
        (1.3537593199999995, 1.3324275399999996, 103.79295100109655, 103.7717563, 35),
        (1.3537593199999995, 1.3324275399999996, 103.8141457021931, 103.79295100109655, 35),
        (1.3537593199999995, 1.3324275399999996, 103.83534040328965, 103.8141457021931, 35),
        (1.3537593199999995, 1.3324275399999996, 103.8565351043862, 103.83534040328965, 35),
        (1.3537593199999995, 1.3324275399999996, 103.87772980548274, 103.8565351043862, 35),
        (1.3537593199999995, 1.3324275399999996, 103.89892450657929, 103.87772980548274, 35),
        (1.3537593199999995, 1.3324275399999996, 103.92011920767584, 103.89892450657929, 35),
        (1.3750910999999995, 1.3537593199999995, 103.79295100109655, 103.7717563, 35),
        (1.3750910999999995, 1.3537593199999995, 103.8141457021931, 103.79295100109655, 35),
        (1.3750910999999995, 1.3537593199999995, 103.83534040328965, 103.8141457021931, 35),
        (1.3750910999999995, 1.3537593199999995, 103.8565351043862, 103.83534040328965, 35),
        (1.3750910999999995, 1.3537593199999995, 103.87772980548274, 103.8565351043862, 35),
        (1.3750910999999995, 1.3537593199999995, 103.89892450657929, 103.87772980548274, 35),
        (1.3750910999999995, 1.3537593199999995, 103.92011920767584, 103.89892450657929, 35),
    ]

    N, S, E, W = (1.4701, 1.1667, 104.0386, 103.5940)  # Singapore limits

    import pickle

    fname = os.path.join(config.osmnx_cache_folder, "test_pickle.pkl")
    if not os.path.exists(fname):
        ss = time.time()
        graph = ox.graph_from_bbox(N, S, E, W, network_type="drive")
        sprint(time.time() - ss, ox.settings.use_cache)

        ss = time.time()
        with open(fname, "wb") as f:
            pickle.dump(graph, f, protocol=config.pickle_protocol)
        print("Pickle write time: ", time.time() - ss)

    elif os.path.exists(fname):
        ss = time.time()
        with open(fname, "rb") as f:
            graph = pickle.load(f)
        print("Pickle read time: ", time.time() - ss)

    gs_edges, gs_nodes = utils_graph.graph_to_gdfs(graph, edges=True, nodes=False), utils_graph.graph_to_gdfs(
        graph, edges=False, nodes=True
    )

    for count, NSEW in enumerate(list_of_bbox):
        # # scl = Scale(RoadNetwork("Singapore"), 5)
        # # pickle.dump(scl, open('scl.pkl', 'wb'))
        # scl = pickle.load(open('scl.pkl', 'rb'))

        N, S, E, W, _ = NSEW

        # print ("Unoptimised code:", sep="")
        # graph = smart_truncate(
        #     graph,
        #     gs_nodes,
        #     gs_edges,
        #     N,
        #     S,
        #     E,
        #     W,
        #     get_features=False,
        #     get_subgraph=True,
        #     scale=25,
        #     legacy=True,
        # )
        sprint(count)

        print("     Optimised code: ", sep="")
        graph = smart_truncate(
            graph,
            gs_nodes,
            gs_edges,
            N,
            S,
            E,
            W,
            get_features=False,
            get_subgraph=True,
            scale=25,
            legacy=False,
        )
