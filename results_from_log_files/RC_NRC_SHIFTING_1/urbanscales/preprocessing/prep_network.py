import copy
import glob
import os
import shutil
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import shapely.geometry
from shapely.geometry import box
import geopandas as gpd
import networkx
import numpy as np
from matplotlib import pyplot as plt

# from line_profiler_pycharm import profile

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from urbanscales.preprocessing.smart_truncate_gpd import smart_truncate
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import time
import contextily as ctx
import networkx as nx
import osmnx as ox
from geopy.distance import geodesic

# from smartprint import smartprint as sprint
from tqdm import tqdm
import pandas

import config
from urbanscales.io.road_network import RoadNetwork
from urbanscales.preprocessing.tile import Tile

import pickle




# All custom unpicklers are due to SO user Pankaj Saini's answer:  https://stackoverflow.com/a/51397373/3896008
class CustomUnpicklerScale(pickle.Unpickler):
    def find_class(self, module, name):
        """
         Custom unpickler class to handle the loading of Scale objects, ensuring compatibility
         and flexibility with changes in module and class definitions.
         """
        if name == "Scale":
            return Scale
        return super().find_class(module, name)


class Scale:
    def __init__(self, RoadNetwork, scale):
        """
        Initializes a Scale object with a given RoadNetwork and a scale level. Manages the creation
        or loading of scale-related data from a pickle file, setting up graph truncation and tiling
        based on configuration settings.
        """
        try:
            fname = os.path.join(
                config.BASE_FOLDER, config.network_folder, RoadNetwork.city_name, "_scale_" + str(scale) + ".pkl"
            )
        except AttributeError as e:
            print (RoadNetwork)
            raise Exception(e)
        if config.scl_delete_existing_pickle_objects:
            if os.path.exists(fname):
                os.remove(fname)

        if os.path.exists(fname):
            with open(fname, "rb") as f:
                ss = time.time()
                temp = copy.deepcopy(pickle.load(f))
                self.__dict__.update(temp.__dict__)
                print("Scale pickle loading time: ", time.time() - ss)
        else:
            self.RoadNetwork = RoadNetwork
            self.__class__.global_G_OSM = self.RoadNetwork.G_osm
            self.scale = scale
            self.tile_area = (config.rn_square_from_city_centre**2) / (scale**2)

            # the following assert ensures that the scale * scale gives the correct number of tiles
            # as this is valid only if our city is a square
            assert config.rn_square_from_city_centre != -1 and config.rn_percentage_of_city_area == 100

            self.set_bbox_sub_G_map()
        if 1==2 and config.MASTER_VISUALISE_EACH_STEP:
            self.visualise()

    def create_subgraphs_from_bboxes_optimised(self, G, bboxes):
        """
        Creates subgraphs from specified bounding boxes, optimized for performance. It involves
        spatial joins and graph operations to extract relevant portions of the graph.
        """
        print ("Converting graph to GDF's")
        starttime = time.time()
        gdf_nodes, _ = ox.graph_to_gdfs(G)
        print("Done .. Converting graph to GDF's completed in: ", round(time.time() - starttime, 2), "seconds")

        bbox_gdf = gpd.GeoDataFrame({'bbox': bboxes},
                                    geometry=[box(west, south, east, north) for north, south, east, west, _ in bboxes],
                                    crs=gdf_nodes.crs)
                                    # After NSEW, the underscore _ (the fifth parameter is total count;
                                    # not needed for this function)

        print("Performing spatial join")
        starttime = time.time()
        joined_nodes = gpd.sjoin(gdf_nodes, bbox_gdf, how='left', predicate='within')
        print ("Spatial join complete in ", round(time.time() - starttime, 2), " seconds")

        subgraphs = {}
        for bbox in tqdm(bbox_gdf['bbox'], desc="Iterating over bboxes"):
            nodes_in_bbox = joined_nodes[joined_nodes['bbox'] == bbox].index

            # Collect edges by checking neighbors of each node in the bbox
            edges_in_bbox = set()
            for node in nodes_in_bbox:
                for neighbor in G.neighbors(node):
                    if G.has_edge(node, neighbor):
                        key = 0 if not G.is_multigraph() else min(G[node][neighbor])
                        edges_in_bbox.add((node, neighbor, key))

            # Create subgraph based on these edges
            G_sub = G.edge_subgraph(edges_in_bbox).copy()

            subgraphs[bbox] = G_sub
            # subgraphs[bbox] = Tile(G_sub, self.tile_area)

        return subgraphs

    def create_subgraphs_from_bboxes(self, G, bboxes):
        """
        Creates subgraphs from bounding boxes without optimization. Directly processes the graph
        to extract subgraphs corresponding to the bounding boxes.
        """
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

        bbox_gdf = gpd.GeoDataFrame({'bbox_id': range(len(bboxes))},
                                    geometry=[box(west, south, east, north) for north, south, east, west, _ in bboxes],
                                    crs=gdf_nodes.crs)

        joined_nodes = gpd.sjoin(gdf_nodes, bbox_gdf, how='left', predicate='within')

        subgraphs = {}
        for bbox_id in tqdm(bbox_gdf['bbox_id'], desc="Iterating over bboxes"):
            nodes_in_bbox = joined_nodes[joined_nodes['bbox_id'] == bbox_id].index

            # Filter edges where at least one node is in the bbox
            # Include edge keys for MultiDiGraph
            edges_in_bbox = [(u, v, k) for u, v, k in G.edges(keys=True) if u in nodes_in_bbox or v in nodes_in_bbox]

            # Create subgraph based on these edges
            G_sub = G.edge_subgraph(edges_in_bbox).copy()

            # G_sub = G.subgraph(nodes_in_bbox).copy()
            subgraphs[bbox_id] = G_sub

        print ("Subgraphs computed; computing tiles now")

        return subgraphs

    # @profile
    def set_bbox_sub_G_map(self, save_to_pickle=True):
        """
        Defines the mapping from bounding boxes to subgraphs and manages the persistence of these mappings
        through pickle files if required.
        """
        fname = os.path.join(
            config.BASE_FOLDER, config.network_folder, self.RoadNetwork.city_name, "_scale_" + str(self.scale) + ".pkl"
        )
        if os.path.exists(fname):
            # do nothing
            return
        self._set_list_of_bbox()
        self.dict_bbox_to_subgraph = {}

        inputs = list(range(len(self.list_of_bbox)))
        starttime = time.time()
        if not os.path.exists(config.warnings_folder):
            os.mkdir(config.warnings_folder)

        if config.ppl_parallel_overall > 1:
            if config.scl_n_jobs_parallel > 1:
                print("Remove one level of parallelisation")
                raise Exception("AssertionError: daemonic processes are not allowed to have children")

        if config.scl_n_jobs_parallel > 1:
            print("Processing in parallel: ")

            print("File counter running in background; clearing temp files")
            print("Cleaning temp files folder")
            try:
                os.mkdir(os.path.join(config.BASE_FOLDER, "temp"))
                os.mkdir(os.path.join(config.BASE_FOLDER, "temp", self.RoadNetwork.city_name))
            except:
                print(
                    "temp Folder for city",
                    self.RoadNetwork.city_name,
                    "already exists! .. continuing to \
                emptying the temp folder",
                )
            shutil.rmtree(os.path.join(config.BASE_FOLDER, "temp", self.RoadNetwork.city_name), ignore_errors=True)
            os.mkdir(os.path.join(config.BASE_FOLDER, "temp", self.RoadNetwork.city_name))
            print("Cleaned the temp folder")


            if config.rn_truncate_method in ["GPD_DUMMY_NODES_SMART_TRUNC", "OSMNX_RETAIN_EDGE"]:
                stop_background_thread = threading.Event()

                self.keep_countin_state = True
                filecounter = threading.Thread(target=self.keep_counting)
                filecounter.start()
                with Pool(config.scl_n_jobs_parallel) as p:
                    list_of_tuples = p.map(self._helper_create_dict_in_parallel, list(self.list_of_bbox))

                self.keep_countin_state = False
                stop_background_thread.clear()

            elif config.rn_truncate_method == "GPD_CUSTOM":
                dict_of_subgraphs = self.create_subgraphs_from_bboxes_optimised(self.RoadNetwork.G_osm, self.list_of_bbox)
                self.dict_bbox_to_subgraph = {}
                counter_bbox = 0
                for bbox in tqdm(dict_of_subgraphs, desc="Converting subgraphs to Tiles"):
                    N,S,E,W, _ = bbox
                    # After NSEW, the underscore _ (the fifth parameter is total count;
                    # not needed for this function)
                    print (len(list(dict_of_subgraphs[bbox].nodes)))
                    try:
                        self.dict_bbox_to_subgraph[N, S, E, W] = Tile(dict_of_subgraphs[bbox], self.tile_area)
                        print (self.dict_bbox_to_subgraph[N, S, E, W].get_features())
                    except Exception as e:
                        self.dict_bbox_to_subgraph[N, S, E, W] = config.rn_no_stats_marker
                        # raise Exception(e)

                    if len(list(dict_of_subgraphs[bbox].nodes())) == 0 or len(list(dict_of_subgraphs[bbox].edges())) == 0:
                        self.dict_bbox_to_subgraph[N, S, E, W] =  config.rn_no_stats_marker

                    counter_bbox += 1

                    debug_stop = True
                    if counter_bbox % 100 == 0 and config.MASTER_VISUALISE_EACH_STEP_INSIDE_PrepNetwork_class:
                        # Plot the bboxes from scl_jf
                        # Example list of bounding boxes
                        # bboxes = list(scl_jf.bbox_segment_map.keys())
                        bboxes = list(self.dict_bbox_to_subgraph.keys())
                        from shapely.geometry import Polygon
                        # Create a GeoDataFrame with these bounding boxes
                        gdf = gpd.GeoDataFrame({
                            'geometry': [Polygon([(lon1, lat1), (lon1, lat2), (lon2, lat2), (lon2, lat1)]) for
                                         lat1, lat2, lon1, lon2 in bboxes]
                        }, crs="EPSG:4326")  # EPSG:4326 is WGS84 latitude-longitude projection

                        # Convert the GeoDataFrame to the Web Mercator projection (used by most web maps)
                        gdf_mercator = gdf.to_crs(epsg=3857)

                        # Plotting
                        fig, ax = plt.subplots(figsize=(10, 10))
                        gdf_mercator.boundary.plot(ax=ax, color='red')
                        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                        ax.set_axis_off()
                        if not os.path.exists(
                                os.path.join(config.BASE_FOLDER, config.network_folder, self.RoadNetwork.city_name,
                                             "OSM_bboxes")):
                            os.mkdir(
                                os.path.join(config.BASE_FOLDER, config.network_folder, self.RoadNetwork.city_name,
                                             "OSM_bboxes"))
                        plt.savefig(
                            os.path.join(config.BASE_FOLDER, config.network_folder, self.RoadNetwork.city_name,
                                         "OSM_bboxes",
                                         f"{self.scale}_bbox_counter_{counter_bbox}.png"))
                        plt.show(block=False)

                    G_sub = dict_of_subgraphs[bbox]
                    if config.MASTER_VISUALISE_EACH_STEP_INSIDE_PrepNetwork_class and G_sub != config.rn_no_stats_marker \
                            and len(G_sub.nodes()) != 0 and len(G_sub.edges()) != 0:
                        # Plot the bboxes from scl_jf
                        # Example list of bounding boxes
                        # bboxes = list(scl_jf.bbox_segment_map.keys())
                        bboxes = [list(self.dict_bbox_to_subgraph.keys())[counter_bbox-1]]
                        from shapely.geometry import Polygon
                        # Create a GeoDataFrame with these bounding boxes
                        gdf = gpd.GeoDataFrame({
                            'geometry': [Polygon([(lon1, lat1), (lon1, lat2), (lon2, lat2), (lon2, lat1)]) for
                                         lat1, lat2, lon1, lon2 in bboxes]
                        }, crs="EPSG:4326")  # EPSG:4326 is WGS84 latitude-longitude projection

                        # Convert the GeoDataFrame to the Web Mercator projection (used by most web maps)
                        gdf_mercator = gdf.to_crs(epsg=3857)

                        # Plotting
                        fig, ax = plt.subplots(figsize=(10, 10))
                        gdf_mercator.boundary.plot(ax=ax, color='red')
                        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

                        G_sub = dict_of_subgraphs[bbox]
                        # ox.plot_graph(G_sub, ax=ax, show=False, edge_color='black')
                        ax.set_axis_off()
                        if not os.path.exists(
                                os.path.join(config.BASE_FOLDER, config.network_folder, self.RoadNetwork.city_name,
                                             "OSM_subgraphs")):
                            os.mkdir(
                                os.path.join(config.BASE_FOLDER, config.network_folder, self.RoadNetwork.city_name,
                                             "OSM_subgraphs"))
                        plt.savefig(
                            os.path.join(config.BASE_FOLDER, config.network_folder, self.RoadNetwork.city_name,
                                         "OSM_subgraphs",
                                         f"{self.scale}_bbox_counter_{counter_bbox}.png"))
                        plt.show(block=False)


                        bboxes = [list(self.dict_bbox_to_subgraph.keys())[counter_bbox-1]]
                        from shapely.geometry import Polygon
                        # Create a GeoDataFrame with these bounding boxes
                        gdf = gpd.GeoDataFrame({
                            'geometry': [Polygon([(lon1, lat1), (lon1, lat2), (lon2, lat2), (lon2, lat1)]) for
                                         lat1, lat2, lon1, lon2 in bboxes]
                        }, crs="EPSG:4326")  # EPSG:4326 is WGS84 latitude-longitude projection

                        # Convert the GeoDataFrame to the Web Mercator projection (used by most web maps)
                        gdf_mercator = gdf.to_crs(epsg=3857)

                        # Plotting
                        fig, ax = plt.subplots(figsize=(10, 10))
                        gdf_mercator.boundary.plot(ax=ax, color='red')
                        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

                        G_sub = dict_of_subgraphs[bbox]
                        ox.plot_graph(G_sub, ax=ax, show=False, edge_color='black')
                        ax.set_axis_off()
                        if not os.path.exists(
                                os.path.join(config.BASE_FOLDER, config.network_folder, self.RoadNetwork.city_name,
                                             "OSM_subgraphs")):
                            os.mkdir(
                                os.path.join(config.BASE_FOLDER, config.network_folder, self.RoadNetwork.city_name,
                                             "OSM_subgraphs"))
                        plt.savefig(
                            os.path.join(config.BASE_FOLDER, config.network_folder, self.RoadNetwork.city_name,
                                         "OSM_subgraphs",
                                         f"{self.scale}_bbox_counter_{counter_bbox}_Subgraph.png"))
                        plt.show(block=False)



                debug_pitstop = True

                # Call the function and pass necessary arguments
                # this is in fact slower since there appears to be multiple parallelisation
                # attemps concurrently in the pipeline
                # self.dict_bbox_to_subgraph = self.process_subgraphs(dict_of_subgraphs, self.tile_area)
            else:
                raise Exception ("Unknown config.rn_truncate_method in prep_network.py")



        elif config.scl_n_jobs_parallel == 1:
            # single threaded
            list_of_tuples = []
            for i in tqdm(
                range(len(inputs)),
                desc="Single threaded performance " + self.RoadNetwork.city_name + str(self.scale),
            ):  # input_ in inputs:
                input_ = self.list_of_bbox[i]
                k, v = self._helper_create_dict_in_parallel(input_)
                list_of_tuples.append((k, v))
        else:
            raise Exception("Wrong number of threads specified in config file.")

        if config.rn_truncate_method != "GPD_CUSTOM":
            # For other cases, we need this. for GPD_custom, this is already saved above directly
            # into the dictionary
            self.dict_bbox_to_subgraph = dict(list_of_tuples)



        print(time.time() - starttime, "seconds using", config.scl_n_jobs_parallel, "threads")

        print("Before removing empty bboxes", len(self.list_of_bbox))

        empty_bboxes = []
        for key in self.dict_bbox_to_subgraph:
            if self.dict_bbox_to_subgraph[key] == config.rn_no_stats_marker:
                empty_bboxes.append(key)

        for key in empty_bboxes:
            del self.dict_bbox_to_subgraph[key]

        self.list_of_bbox = list(set(self.list_of_bbox) - set(empty_bboxes))
        print("After removing empty bboxes", len(self.list_of_bbox))

        if not os.path.exists(fname):
            rand_pickle_marker = os.path.join(config.temp_folder_for_robust_pickle_files,
                                              str(int(np.random.rand() * 100000000000000)))
            with open(rand_pickle_marker, "wb") as f:
                pickle.dump(self, f, protocol=config.pickle_protocol)
            os.rename(rand_pickle_marker, fname)

    def process_bbox(self, bbox, dict_of_subgraphs, tile_area):
        """
        Processes a single bounding box to generate a Tile object based on the subgraph extracted for
        that bounding box.
        """
        N, S, E, W, _ = bbox
        try:
            tile = Tile(dict_of_subgraphs[bbox], tile_area)
            return (N, S, E, W), tile
        except Exception as e:
            return (N, S, E, W), config.rn_no_stats_marker

    def process_subgraphs(self, dict_of_subgraphs, tile_area):
        """
        Processes all subgraphs using parallel processing to convert them into Tile objects and compiles
        the results into a dictionary mapping bounding boxes to Tiles.
        """
        results = []
        with ProcessPoolExecutor(max_workers=config.scl_n_jobs_parallel) as executor:
            # Submit tasks
            future_to_bbox = {executor.submit(self.process_bbox, bbox, dict_of_subgraphs, tile_area): bbox for bbox in
                              dict_of_subgraphs}

            # Process results with a progress bar
            with tqdm(total=len(future_to_bbox), desc="Processing Subgraphs") as progress:
                for future in as_completed(future_to_bbox):
                    result = future.result()
                    results.append(result)
                    progress.update(1)

        # Combine the results into a dictionary
        dict_bbox_to_subgraph = {bbox: tile for bbox, tile in results}
        return dict_bbox_to_subgraph

    def _set_list_of_bbox(self):
        """
        Generates a list of bounding boxes based on the scale and configuration settings of the RoadNetwork.
        """
        self.list_of_bbox = []

        # X: lon
        # Y: Lat

        self._helper_compute_deltas()

        srn = self.RoadNetwork

        start_y = srn.min_y
        num_tiles = int((srn.max_y - srn.min_y) / self.delta_y * int((srn.max_x - srn.min_x) / self.delta_x))
        pbar = tqdm(total=num_tiles, desc="Creating list of bboxes.. ")

        while start_y + self.delta_y <= srn.max_y:
            start_x = srn.min_x
            while start_x + self.delta_x <= srn.max_x:
                # bbox_poly = geometry.box(start_x, start_x + self.delta_x, start_y, start_y + self.delta_y, ccw=True)

                N = start_y + self.delta_y
                S = start_y
                E = start_x + self.delta_x
                W = start_x
                # self.dict_bbox_to_subgraph[(N,S,E,W)] = ox.truncate.truncate_graph_bbox(srn.G_osm, N, S, E, W)
                self.list_of_bbox.append([N, S, E, W])
                pbar.update(1)

                start_x += self.delta_x

            start_y += self.delta_y

        debug_stop = 2
        len_ = len(self.list_of_bbox)
        for i in range(len_):
            self.list_of_bbox[i] = tuple(self.list_of_bbox[i] + [len_])

    def _helper_compute_deltas(self):
        """
        Helper function to compute delta values for latitude and longitude based on the scale and dimensions
        of the road network.
        """
        srn = self.RoadNetwork

        # to take care of negative (southern hemisphere cases)
        if srn.max_y < srn.min_y:
            srn.max_y, srn.min_y = srn.min_y, srn.max_y

        if srn.max_x < srn.min_x:
            srn.max_x, srn.min_x = srn.min_x, srn.max_x

        assert srn.max_y > srn.min_y
        assert srn.max_x > srn.min_x

        diagonal = geodesic((srn.max_y, srn.max_x), ((srn.min_y, srn.min_x))).meters
        y_edge_1 = geodesic((srn.max_y, srn.max_x), ((srn.min_y, srn.max_x))).meters
        x_edge_1 = geodesic((srn.max_y, srn.max_x), ((srn.max_y, srn.min_x))).meters
        y_edge_2 = geodesic((srn.min_y, srn.max_x), ((srn.max_y, srn.max_x))).meters
        x_edge_2 = geodesic((srn.min_y, srn.max_x), ((srn.min_y, srn.min_x))).meters

        # assert < 0.1 % error in distance computation
        err = config.scl_error_percentage_tolerance
        try:
            assert (((y_edge_1**2 + x_edge_1**2) ** 0.5 - diagonal) / diagonal * 100) < err
            assert ((y_edge_1 - y_edge_2) / y_edge_1 * 100) < err
            assert ((x_edge_1 - x_edge_2) / x_edge_1 * 100) < err
        except:
            print((((y_edge_1**2 + x_edge_1**2) ** 0.5 - diagonal) / diagonal * 100))
            print((y_edge_1 - y_edge_2) / y_edge_1 * 100)
            print((x_edge_1 - x_edge_2) / x_edge_1 * 100)
            raise Exception("Error in _helper_compute_deltas")
            # sys.exit(0)

        self.x_edge = (x_edge_1 + x_edge_2) / 2
        self.y_edge = (y_edge_1 + y_edge_2) / 2
        self.aspect_y_by_x = self.y_edge / self.x_edge

        self.delta_y = (srn.max_y - srn.min_y) / self.scale
        self.delta_x = (srn.max_x - srn.min_x) / self.scale * self.aspect_y_by_x

    def create_file_marker(self):
        """
        Creates a temporary marker file used in parallel processing to track progress and manage state.
        """
        with open(
            os.path.join(
                config.BASE_FOLDER,
                "temp",
                self.RoadNetwork.city_name,
                (
                    "temp-"
                    + self.RoadNetwork.city_name
                    + str(self.scale)
                    + "-"
                    + str(int(np.random.rand() * 10000000))
                    + ".marker"
                ),
            ),
            "w",
        ) as f:
            f.write("Done")

    def keep_counting(self):
        """
        Background process to count and update the progress of file processing during parallel execution.
        """
        oldcount = 0
        with tqdm(total=self.list_of_bbox[0][-1], desc="Counting files ") as pbar:
            while self.keep_countin_state:
                count = len(
                    glob.glob(
                        os.path.join(
                            config.BASE_FOLDER,
                            "temp",
                            self.RoadNetwork.city_name,
                            ("temp-" + self.RoadNetwork.city_name + str(self.scale) + "*.marker"),
                        )
                    )
                )
                assert self.list_of_bbox[0][-1] == len(self.list_of_bbox)
                if config.scl_temp_file_counter:
                    pbar.update(count - oldcount)
                time.sleep(1)
                oldcount = count

    # @profile
    def _helper_create_dict_in_parallel(self, key):
        """
        Helper function to process a single tile or bounding box in parallel, handling graph truncation
        and tile creation.
        """
        if config.scl_temp_file_counter:
            self.create_file_marker()

        if config.rn_truncate_method in ["OSMNX_RETAIN_EDGE", "GPD_DUMMY_NODES_SMART_TRUNC" ]:
            N, S, E, W, total = key

            from shapely.geometry import box as bboxShapely
            def do_not_overlap(N, S, E, W, N_JF_data, S_JF_data, E_JF_data, W_JF_data):
                """
                Checks if two bounding boxes overlap. This function ensures that graph processing occurs only for areas
                that have relevant data by comparing geographic boundaries.
                """
                bbox1 = bboxShapely(W, S, E, N)
                bbox2 = bboxShapely(W_JF_data, S_JF_data, E_JF_data, N_JF_data)
                return not bbox1.intersects(bbox2)

            def debug_by_plotting_bboxes(color, small_G=None):
                """
                Provides a debugging visualization tool that plots bounding boxes with optional graph data to identify
                and diagnose issues with graph processing based on geographic boundaries.
                """
                fig, ax = ox.plot.plot_graph(
                    self.RoadNetwork.G_osm,
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

                rect = plt.Rectangle((W, S), E - W, N - S, facecolor=color, alpha=0.3, edgecolor=None)
                ax.add_patch(rect)
                plt.savefig(
                    os.path.join(config.BASE_FOLDER, config.results_folder, "empty_tiles", self.RoadNetwork.city_name)
                    + str(self.scale)
                    + color
                    + str(int(np.random.rand() * 100000000))
                    + ".png"
                )
                # plt.show(block=False)


            # the format is as shown below: (copied from config file)
            # rn_city_wise_bboxes = {
            #     "Singapore": [1.51316, 104.135278, 1.130361, 103.566667],

            N_JF_data,  E_JF_data, S_JF_data, W_JF_data = config.rn_city_wise_bboxes[self.RoadNetwork.city_name]
            if do_not_overlap(N, S, E, W, N_JF_data, S_JF_data, E_JF_data, W_JF_data) :
                print ("Subgraph removed as empty, since no overlap found")
                return (key, config.rn_no_stats_marker) # no need to process these graphs if we don't have their speed data
            else:
                try:
                    if config.rn_truncate_method == "GPD_DUMMY_NODES_SMART_TRUNC":
                        truncated_graph = smart_truncate(
                            self.RoadNetwork.G_osm, self.RoadNetwork.G_OSM_nodes, self.RoadNetwork.G_OSM_edges, N, S, E, W
                        )
                    elif config.rn_truncate_method == "OSMNX_RETAIN_EDGE":
                        # truncated_graph = ox.truncate.truncate_graph_bbox(self.RoadNetwork.G_osm, N,S,E,W, truncate_by_edge=True)
                        truncated_graph = ox.truncate.truncate_graph_bbox(self.__class__.global_G_OSM, N,S,E,W, truncate_by_edge=True)
                    else:
                        raise Exception ("Wrong Truncation method passed!")

                    if not isinstance(truncated_graph, networkx.MultiDiGraph):
                        if truncated_graph == config.rn_no_stats_marker:
                            raise ValueError
                        else:
                            raise Exception("Unknown Error; Not Null passed")

                    tile = Tile(truncated_graph, self.tile_area)
                    # if config.verbose >= 2:
                    #     with open(os.path.join(config.warnings_folder, "empty_graph_tiles.txt"), "a") as f:
                    #         csvwriter = csv.writer(f)
                    #         csvwriter.writerow(["ValueError at i: " + str(i) + " " + self.RoadNetwork.city_name])
                    if config.DEBUG:
                        debug_by_plotting_bboxes("green", tile.G)
                except ValueError:  #
                    # if config.verbose >= 1:
                    #     with open(os.path.join(config.warnings_folder, "empty_graph_tiles.txt"), "a") as f:
                    #         csvwriter = csv.writer(f)
                    #         csvwriter.writerow(["ValueError at i: " + str(i) + " " + self.RoadNetwork.city_name])
                    if config.DEBUG:
                        debug_by_plotting_bboxes("red")
                    return (key, config.rn_no_stats_marker)
                except nx.exception.NetworkXPointlessConcept:
                    if config.DEBUG:
                        debug_by_plotting_bboxes("blue")
                    return (key, config.rn_no_stats_marker)
                return (key, tile)



            def create_subgraph_within_bbox(bbox):
                """
                Creates a subgraph from the main road network graph within the specified bounding box. This function is
                utilized in the context of preparing smaller, manageable sections of a larger road network for detailed analysis.
                """
                north, south, east, west = bbox

                # Identify nodes within the bounding box
                startime = time.time()
                nodes_in_bbox = self.RoadNetwork.G_OSM_nodes.cx[west:east, south:north]
                # nodes_in_bbox = gdf_nodes.cx[west:east, south:north]
                print (time.time() - startime, " Extracting nodes from big graph")

                # Initialize a set with these nodes
                nodes_set = set(nodes_in_bbox.index)

                # Include neighbors of these nodes
                for node in nodes_in_bbox.index:
                    neighbors = set(self.RoadNetwork.G_osm.neighbors(node))
                    # neighbors = set(G_osm_.neighbors(node))
                    nodes_set.update(neighbors)

                # Create and return the subgraph
                startime = time.time()
                G_sub = self.RoadNetwork.G_osm.subgraph(nodes_set).copy()
                # G_sub = G_osm_.subgraph(nodes_set).copy()
                print ("\n Copying the subgraph", time.time() - startime)

                return G_sub

            N, S, E, W, _ = key
            # N, S, E, W, total, extras = key
            # G_osm_, gdf_nodes = extras

            if self.RoadNetwork.G_OSM_nodes.shape[0] == 0:
                print ("Empty tile; returning empty tile marker")
                # return (key, config.rn_no_stats_marker)
                fname = os.path.join(config.BASE_FOLDER, "_prep_network_temp_file_" + str(int(np.random.rand() * 10000000000)) + self.RoadNetwork.city + str(self.RoadNetwork.scale) + ".pkl")
                rand_pickle_marker = os.path.join(config.temp_folder_for_robust_pickle_files,
                                                  str(int(np.random.rand() * 100000000000000)))
                with open(rand_pickle_marker, "wb") as f:
                    pickle.dump((N, S, E, W, config.rn_no_stats_marker), f, protocol=config.pickle_protocol)
                os.rename(rand_pickle_marker, fname)

            try:
                startime = time.time()
                tile = Tile(create_subgraph_within_bbox(bbox=(N, S, E, W)), self.tile_area)
                print(time.time() - startime, "Computing tile features")

            except:
                print ("Error in tile computation; returning empty tile marker")
                tile = config.rn_no_stats_marker

            # return (key, tile)
            fname = os.path.join(config.BASE_FOLDER, "_prep_network_temp_file_" + str(
                int(np.random.rand() * 10000000000)) + self.RoadNetwork.city + str(self.RoadNetwork.scale) + ".pkl")
            rand_pickle_marker = os.path.join(config.temp_folder_for_robust_pickle_files,
                                              str(int(np.random.rand() * 100000000000000)))
            with open(rand_pickle_marker, "wb") as f:
                pickle.dump((N, S, E, W, tile), f, protocol=config.pickle_protocol)
            os.rename(rand_pickle_marker, fname)



    def visualise(self):
        """
        Generates a visualization of bounding boxes processed in the Scale, overlaying them on a basemap.
        """
        identifier = self.RoadNetwork.city_name + "_" + str(self.scale) + "_" + str(config.shift_tile_marker)
        # Ensure that dict_bbox_to_subgraph attribute exists
        if not hasattr(self, 'dict_bbox_to_subgraph'):
            print("No dict_bbox_to_subgraph attribute found in the scale object.")
            return

        # Create a list of box geometries, excluding those with rn_no_stats_marker
        geometries = []
        for bbox in self.dict_bbox_to_subgraph:
            if self.dict_bbox_to_subgraph[bbox] != config.rn_no_stats_marker:
                N, S, E, W = bbox
                geometries.append(box(W, S, E, N))

        if not geometries:
            print("No valid bounding boxes to plot.")
            return

        # Create GeoDataFrame from the geometries
        gdf = gpd.GeoDataFrame({'geometry': geometries}, crs='EPSG:4326')

        # Convert to Web Mercator for basemap
        gdf = gdf.to_crs(epsg=3857)
        fig, ax = plt.subplots(figsize=(15, 10))
        gdf.boundary.plot(ax=ax, color='blue', linewidth=0.5, alpha=0.5)
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik) #, zoom=config.scl_basemap_zoom_level)
        plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, self.RoadNetwork.city_name,f"Bboxes_{identifier}.png"), dpi=600)
        plt.show(block=False)

    # Example usage
    # scale_obj = Scale(RoadNetwork('CityName'), scale)
    # plot_bboxes_for_debugging(scale_obj, 'CityName_Scale')


    @staticmethod
    def generate_scales_for_all_cities():
        """
        Generates scale objects for all cities as per configuration, useful for batch processing across multiple scales.
        """
        for city in config.scl_master_list_of_cities:
            for seed in config.scl_list_of_seeds:
                for depth in config.scl_list_of_depths:
                    Scale(RoadNetwork(city), seed**depth)
                    print(city, seed, depth)
                    loaded_scale = Scale.get_object(cityname=city, scale=seed**depth)

    @staticmethod
    def get_object(cityname, scale):
        """
        Retrieves a saved Scale object from disk using the specified city name and scale, handling
        file access and custom unpickling.
        """

        fname = os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "_scale_" + str(scale) + ".pkl")
        if os.path.exists(fname):
            obj = CustomUnpicklerScale(open(fname, "rb")).load()
        else:
            raise Exception(
                fname
                + " not present \n Run prep_network.py, speed_data.py and prep_speed.py before running this function"
            )
            # sys.exit(0)
        return obj


    def __repr__(self):
        """
        Return a string representation of the Scale object that includes all relevant attributes
        for debugging and object state reproduction.
        """
        return (f"Scale(RoadNetwork={self.RoadNetwork.__repr__()}, "
                f"scale={self.scale}, "
                f"global_G_OSM={'Loaded' if self.global_G_OSM else 'None'}, "
                f"tile_area={self.tile_area}, "
                f"list_of_bbox={self.list_of_bbox}, "
                f"dict_bbox_to_subgraph=Length {len(self.dict_bbox_to_subgraph)})")



if __name__ == "__main__":
    Scale.generate_scales_for_all_cities()
    debug_stop = 1
