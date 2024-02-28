import os
import sys
from multiprocessing import Pool

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


import shapely.wkt
import numpy as np
import shapely.ops
import config
from urbanscales.preprocessing.prep_network import Scale
from urbanscales.io.road_network import RoadNetwork

# Segment is greyed out in IDEs like PyCharm hinting that the import is not being used but it is needed for pickle loadings
from urbanscales.io.speed_data import SpeedData, Segment

import os
import copy
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import box

import pandas as pd
import pandas

from smartprint import smartprint as sprint
import shapely.geometry

import pickle


# All custom unpicklers are due to SO user Pankaj Saini's answer:  https://stackoverflow.com/a/51397373/3896008
class CustomUnpicklerScaleJF(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "ScaleJF":
            return ScaleJF
        if name == "Scale":
            from urbanscales.preprocessing.prep_network import Scale
        if name == "RoadNetwork":
            from urbanscales.io.road_network import RoadNetwork
            return RoadNetwork
        return super().find_class(module, name)


class ScaleJF:
    """
    Attributes:
        self.bbox_segment_map = {}
        self.bbox_jf_map = {}
        self.Scale = scale
        self.SpeedData = speed_data
    """

    def __init__(self, scale: Scale, speed_data: SpeedData, tod: int):
        assert scale.RoadNetwork.city_name == speed_data.city_name
        fname = os.path.join(
            config.BASE_FOLDER,
            config.network_folder,
            scale.RoadNetwork.city_name,
            "_scale_" + str(scale.scale) + "_prep_speed_" + str(tod) + ".pkl",
        )
        if config.ps_delete_existing_pickle_objects:
            if os.path.exists(fname):
                os.remove(fname)

        if os.path.exists(fname):
            with open(fname, "rb") as f:
                temp = copy.deepcopy(pickle.load(f))
                self.__dict__.update(temp.__dict__)
            print ("Prep-speed Pickle object loaded for (city, scale, tod): ", (scale.RoadNetwork.city_name, scale.scale, tod))
        else:
            sprint (fname)
            print("Prep-speed Pickle Not found for (city, scale, tod): ", (scale.RoadNetwork.city_name, scale.scale, tod))
            self.bbox_segment_map = {}
            self.bbox_jf_map = {}
            self.Scale = scale
            self.SpeedData = speed_data
            self.tod = tod
            self.set_bbox_segment_map()
            self.set_bbox_jf_map()


    def set_bbox_segment_map(self):
        fname = os.path.join(
            config.BASE_FOLDER,
            config.network_folder,
            self.Scale.RoadNetwork.city_name,
            "_scale_" + str(self.Scale.scale) + "_prep_speed_" + str(self.tod) + ".pkl",
        )

        if os.path.exists(fname):
            self.bbox_segment_map = copy.deepcopy(
                (ScaleJF.get_object(self.Scale.RoadNetwork.city_name, self.Scale.scale, self.tod)).bbox_segment_map,
            )
            return

        else:
            for tod_ in config.td_tod_list:
                fnametemp = os.path.join(
                    config.BASE_FOLDER,
                    config.network_folder,
                    self.Scale.RoadNetwork.city_name,
                    "_scale_" + str(self.Scale.scale) + "_prep_speed_" + str(tod_) + ".pkl",
                )
                if os.path.exists(fnametemp):
                    print("\n", fnametemp, " found; re-using bbox segment map from here")
                    self.bbox_segment_map = copy.deepcopy(
                        (ScaleJF.get_object(self.Scale.RoadNetwork.city_name, self.Scale.scale, tod_)).bbox_segment_map,
                    )
                    return

            print("Could not find other tod's, Recomputing bbox-segment map for this tod: ", self.tod)

            # Convert segments into GeoDataFrame
            segments_gdf = gpd.GeoDataFrame({
                'geometry': [shapely.wkt.loads(str(seg)) for seg in self.SpeedData.segment_jf_map]
            })

            # Convert bounding boxes into GeoDataFrame
            if config.rn_truncate_method != "GPD_CUSTOM":
                bboxes = [
                    box(W, S, E, N) for bbox in self.Scale.dict_bbox_to_subgraph.keys() for N, S, E, W, _unused_len in
                    [bbox]
                ]
            else:
                bboxes = [
                    box(W, S, E, N) for bbox in self.Scale.dict_bbox_to_subgraph.keys() for N, S, E, W in
                    [bbox]
                ]
            bboxes_gdf = gpd.GeoDataFrame({'bbox': self.Scale.dict_bbox_to_subgraph.keys(), 'geometry': bboxes})

            # Spatial join to find intersections
            intersections = gpd.sjoin(segments_gdf, bboxes_gdf, op='intersects')
            self.bbox_segment_map = intersections.groupby('bbox')['geometry'].apply(list).to_dict()

            sprint(len(self.bbox_segment_map))

    def set_bbox_segment_map_loops(self):
        # Step 1: iterate over segments
        # Step 2: iterate over bboxes
        # Then within the loop populate the dict if both are intersecting
        fname = os.path.join(
            config.BASE_FOLDER,
            config.network_folder,
            self.Scale.RoadNetwork.city_name,
            "_scale_" + str(self.Scale.scale) + "_prep_speed_" + str(self.tod) + ".pkl",
        )
        if os.path.exists(fname):
            # @P/A Ask 2
            self.bbox_segment_map = copy.deepcopy(
                (ScaleJF.get_object(self.Scale.RoadNetwork.city_name, self.Scale.scale, self.tod)).bbox_segment_map,
            )
            return

        else:

            for tod_ in config.td_tod_list:
                fnametemp = os.path.join(
                    config.BASE_FOLDER,
                    config.network_folder,
                    self.Scale.RoadNetwork.city_name,
                    "_scale_" + str(self.Scale.scale) + "_prep_speed_" + str(tod_) + ".pkl",
                )
                if os.path.exists(fnametemp):
                    print("\n", fnametemp, " found; re-using bbox segment map from here")
                    self.bbox_segment_map = copy.deepcopy(
                        (ScaleJF.get_object(self.Scale.RoadNetwork.city_name, self.Scale.scale, tod_)).bbox_segment_map,
                    )
                    return  # and continue to creating this new prep_speed_object using fname

            print ("Could not find other tod's, Recomputing bbox-segment map for this tod: ", self.tod)

            count_ = len(self.SpeedData.NID_road_segment_map) * len(self.Scale.dict_bbox_to_subgraph)
            pbar = tqdm(
                total=count_,
                desc="Setting bbox Segment map...",
            )
            for segment in self.SpeedData.segment_jf_map:
                # print("Inside loop 1")
                seg_poly = shapely.wkt.loads(segment)

                for bbox in self.Scale.dict_bbox_to_subgraph.keys():
                    # print("                   Inside loop 2")

                    N, S, E, W, _unused_len = bbox

                    bbox_shapely = shapely.geometry.box(W, S, E, N, ccw=True)
                    if seg_poly.intersection(bbox_shapely):
                        if bbox in self.bbox_segment_map:
                            self.bbox_segment_map[bbox].append(segment)
                        else:
                            self.bbox_segment_map[bbox] = [segment]

                    pbar.update(1)
            sprint(len(self.bbox_segment_map))

    def set_bbox_jf_map(self):
        """
        Input: object of class Scale JF
        For the current tod, this function sets its Y (JF mean/median) depending on config
        """

        # Convert bbox_segment_map to DataFrame for vectorized operations
        bbox_segment_df = pd.DataFrame({
            'bbox': list(self.bbox_segment_map.keys()),
            'segments': [self.bbox_segment_map[k] for k in self.bbox_segment_map.keys()]
        })

        if not config.ps_set_all_speed_zero:
            # Fetch segment_jf values for each segment using map and apply operations
            bbox_segment_df['segment_jf_values'] = bbox_segment_df['segments'].apply(
                lambda segments_list: [self.SpeedData.segment_jf_map.get(str(seg), [np.nan] * 24)[self.tod] for seg in segments_list]
            )

            a = []
            for key in self.SpeedData.segment_jf_map:
                a.append(len(self.SpeedData.segment_jf_map[key]))
            assert len(set(a)) == 1

            # b = []
            # for key in self.SpeedData.segment_jf_map:
            #     a = []
            #     for counter, val in enumerate(self.SpeedData.segment_jf_map[key]):
            #         if counter % self.tod == 0 and self.SpeedData.segment_jf_map[key][counter] != -1:
            #             if self.SpeedData.segment_jf_map[key][counter] == -1:
            #                 print ("-1 Found")
            #             a.append(self.SpeedData.segment_jf_map[key][counter])
            #     b.append(a)
            # bbox_segment_df['segment_jf_values'] = pd.Series(b)

            bbox_segment_df['segment_lengths'] = bbox_segment_df['segments'].apply(
                lambda segments_list: [seg.length for seg in segments_list]
            )

            from shapely.geometry import box

            def segment_length_within_bbox(segments, bbox):
                """
                Calculate the length of each segment within the bounding box.

                Parameters:
                segments (list): List of shapely.geometry.MultiLineString objects representing the segments.
                bbox (tuple): Tuple representing the bounding box (N, S, E, W).

                Returns:
                list: List of lengths of each segment within the bounding box.
                """
                N, S, E, W = bbox
                bbox_polygon = box(W, S, E, N)
                lengths = []

                for segment in segments:
                    intersection = segment.intersection(bbox_polygon)
                    lengths.append(intersection.length)

                return lengths

            # Apply the function to each row of the DataFrame
            bbox_segment_df['lengths_within_bbox'] = bbox_segment_df.apply(
                lambda row: segment_length_within_bbox(row['segments'], row['bbox']), axis=1
            )

            for i in range(bbox_segment_df.shape[0]):
                assert len(bbox_segment_df["lengths_within_bbox"][i]) == len(bbox_segment_df["segments"][i])
                assert len(bbox_segment_df["segment_jf_values"][i]) == len(bbox_segment_df["segments"][i])

            if config.MASTER_VISUALISE_EACH_STEP:
                import matplotlib.pyplot as plt

                for bbox_num in range(bbox_segment_df.shape[0]):

                    if np.random.rand() > 0.99:
                        continue

                    # Extract the segments and bounding box
                    segments = bbox_segment_df['segments'][bbox_num]
                    bbox = bbox_segment_df['bbox'][bbox_num]
                    bbox_polygon = box(bbox[3], bbox[1], bbox[2], bbox[0])

                    # Plot the segments and their intersections with the bounding box
                    fig, ax = plt.subplots()

                    for segment in segments:
                        # Plot each LineString in the MultiLineString
                        for line in segment:
                            x, y = line.xy
                            ax.plot(x, y, color='black', alpha=0.3, linewidth=2, label='Segment outside BBox')

                        # Calculate the intersection of the segment with the bounding box
                        intersection = segment.intersection(bbox_polygon)

                        # Plot the intersection
                        if not intersection.is_empty:
                            if intersection.geom_type == 'MultiLineString':
                                for line in intersection:
                                    x, y = line.xy
                                    ax.plot(x, y, color='black', alpha=1, linewidth=2, label='Segment inside BBox')
                            elif intersection.geom_type == 'LineString':
                                x, y = intersection.xy
                                ax.plot(x, y, color='black', alpha=1, linewidth=2, label='Segment inside BBox')

                    # Plot the bounding box
                    x, y = bbox_polygon.exterior.xy
                    ax.plot(x, y, color='red', alpha=0.7, linewidth=1, linestyle='--', label='Bounding Box')

                    # Set plot properties
                    ax.set_aspect('equal')
                    # plt.legend()
                    plt.title("Segments lengths: " + str(bbox_segment_df["segment_lengths"][bbox_num]) +  " \n"
                                                                                            "Weighted " +
                              str(bbox_segment_df["lengths_within_bbox"][bbox_num]), fontsize=8,
                              )
                    if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, self.Scale.RoadNetwork.city_name, "intersecting_seg_bbox")):
                        os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, self.Scale.RoadNetwork.city_name, "intersecting_seg_bbox"))
                    plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, self.Scale.RoadNetwork.city_name, "intersecting_seg_bbox", str(bbox_num) + ".png"), dpi=300)
                    plt.show()

            # Use np.mean or np.max as aggregation function based on config
            if config.ps_spatial_combination_method == "mean":
                agg_func = np.nanmean
            elif config.ps_spatial_combination_method == "max":
                agg_func = np.nanmax
            elif config.ps_spatial_combination_method == "variance":
                agg_func = np.nanstd
            elif config.ps_spatial_combination_method == "length_weighted_mean":
                agg_func = lambda values, lengths: np.sum(np.multiply(values, lengths)) / np.sum(lengths)
            else:
                raise ValueError(f"Unsupported spatial combination method: {config.ps_spatial_combination_method}")

            if config.ps_spatial_combination_method == "length_weighted_mean":
                bbox_segment_df['agg_jf_value'] = bbox_segment_df.apply(
                    lambda row: agg_func(row['segment_jf_values'], row['lengths_within_bbox']), axis=1)
            elif config.ps_spatial_combination_method in ["mean", "max", "variance"]:
                # Aggregate segment_jf_values for each bbox
                bbox_segment_df['agg_jf_value'] = bbox_segment_df['segment_jf_values'].apply(agg_func)
            else:
                raise ValueError(f"Unsupported spatial combination method: {config.ps_spatial_combination_method}")

            # Update bbox_jf_map from DataFrame
            self.bbox_jf_map = bbox_segment_df.set_index('bbox')['agg_jf_value'].to_dict()

        elif config.ps_set_all_speed_zero:
            # Set all values to zero if ps_set_all_speed_zero is True
            self.bbox_jf_map = {bbox: 0 for bbox in self.bbox_segment_map.keys()}

        # Saving the object
        fname = os.path.join(
            config.BASE_FOLDER,
            config.network_folder,
            self.Scale.RoadNetwork.city_name,
            "_scale_" + str(self.Scale.scale) + "_prep_speed_" + str(self.tod) + ".pkl",
        )
        rand_pickle_marker = os.path.join(config.temp_folder_for_robust_pickle_files,
                                          str(int(np.random.rand() * 100000000000000)))
        with open(rand_pickle_marker, "wb") as f:
            pickle.dump(self, f, protocol=config.pickle_protocol)
            print("Pickle saved!")
        os.rename(rand_pickle_marker, fname)

    def set_bbox_jf_map_loops(self):
        """
        Input: object of class Scale JF
        For the current tod, this function sets its Y (JF mean/median) depending on config
        """

        if not config.ps_set_all_speed_zero:
            for bbox, segments in self.bbox_segment_map.items():
                segment_jf_values = []
                for segment in segments:
                    # Fetch segment_jf value for the segment, and use NaN if not available
                    segment_jf = self.SpeedData.segment_jf_map.get(segment, [np.nan] * 24)[self.tod]
                    segment_jf_values.append(segment_jf)

                # Use mean or max as aggregation function based on config
                if config.ps_spatial_combination_method == "mean":
                    self.bbox_jf_map[bbox] = np.mean(segment_jf_values)
                elif config.ps_spatial_combination_method == "max":
                    self.bbox_jf_map[bbox] = np.max(segment_jf_values)
                else:
                    raise ValueError(f"Unsupported spatial combination method: {config.ps_spatial_combination_method}")

        elif config.ps_set_all_speed_zero:
            # Set all values to zero if ps_set_all_speed_zero is True
            for bbox in self.bbox_segment_map.keys():
                self.bbox_jf_map[bbox] = 0

        # Saving the object
        fname = os.path.join(
            config.BASE_FOLDER,
            config.network_folder,
            self.Scale.RoadNetwork.city_name,
            "_scale_" + str(self.Scale.scale) + "_prep_speed_" + str(self.tod) + ".pkl",
        )
        rand_pickle_marker = os.path.join(config.temp_folder_for_robust_pickle_files,
                                          str(int(np.random.rand() * 100000000000000)))
        with open(rand_pickle_marker, "wb") as f:
            pickle.dump(self, f, protocol=config.pickle_protocol)
            print("Pickle saved!")
        os.rename(rand_pickle_marker, fname)

    def get_object(cityname, scale, tod):
        """
        This function uses the idea that the network and its maps are the same for all tod's
        Hence, it reads the static part (bbox to segment map) from the pickle and then
        create bbox_jf_map for the particular tod of this object
        Returns: (Saved) Object of this class (Scale)

        """
        fname = os.path.join(
            config.BASE_FOLDER,
            config.network_folder,
            cityname,
            "_scale_" + str(scale) + "_prep_speed_" + str(tod) + ".pkl",
        )

        if os.path.exists(fname):
            # ScaleJF.preprocess_different_tods([tod], Scale.get_object_at_scale(cityname, scale), SpeedData.get_object(cityname))
            try:
                obj = CustomUnpicklerScaleJF(open(fname, "rb")).load()
            except EOFError:
                print(fname)
                raise Exception("Error! Corrupted pickle file prep_speed:\n Filename:  " + fname)
        else:
            raise Exception("Error! trying to read prep_speed file that does not exist: Filename: " + fname)

        # obj.tod = tod
        # obj.set_bbox_jf_map()
        return obj

    def preprocess_different_tods(range_element, scl: Scale, sd: SpeedData):
        for tod in tqdm(range_element, desc="Processing for different ToDs"):
            scl_jf = ScaleJF(scl, sd, tod=tod)

    @staticmethod
    def helper_parallel(params):
        city, seed, depth = params
        sd = SpeedData(city, config.sd_raw_speed_data_gran, config.sd_target_speed_data_gran)
        try:
            scl = Scale(RoadNetwork(city), seed**depth)
        except Exception as e:
            sprint(city, seed, depth)
            raise Exception(e)
            # sys.exit(0)
        ScaleJF.preprocess_different_tods(config.ps_tod_list, scl, sd)

    @staticmethod
    def connect_speed_and_nw_data_for_all_cities():
        list_of_parallel_items = []
        for city in config.scl_master_list_of_cities:
            for seed in config.scl_list_of_seeds:
                for depth in config.scl_list_of_depths:
                    list_of_parallel_items.append((city, seed, depth))

        if config.ppl_parallel_overall > 1:
            p = Pool(config.ppl_parallel_overall)
            print(p.map(ScaleJF.helper_parallel, list_of_parallel_items))
        else:
            for params in list_of_parallel_items:
                ScaleJF.helper_parallel(params)

    def __repr__(self):
        return f"ScaleJF(scale={self.Scale.scale}, city_name={self.Scale.RoadNetwork.city_name}, tod={self.tod})"


if __name__ == "__main__":
    sys.path.append(config.home_folder_path)
    ScaleJF.connect_speed_and_nw_data_for_all_cities()
    debug_stop = 2
