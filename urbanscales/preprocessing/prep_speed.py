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
            if isinstance(self.tod, int):
                segment_jf_values = []

                for segments_list in bbox_segment_df['segments']:
                    jf_values_list = []
                    for seg in segments_list:
                        jf_values = self.SpeedData.segment_jf_map.get(str(seg), [np.nan] * 24)[self.tod]
                        jf_values_list.append(jf_values)
                    segment_jf_values.append(jf_values_list)

                bbox_segment_df['segment_jf_values'] = segment_jf_values # one value for each segment, multiple values for each bbox

            if isinstance(self.tod, list): # list of integers like [6,7,8,9]
                segment_jf_values = []

                for segments_list in bbox_segment_df['segments']:
                    jf_values_list = []
                    for seg in segments_list:
                        if config.sd_temporal_combination_method == "mean":
                            agg_func = np.nanmean
                        elif config.sd_temporal_combination_method == "max":
                            agg_func = np.nanmax
                        elif config.sd_temporal_combination_method == "variance":
                            agg_func = np.nanstd
                        jf_values = agg_func( self.SpeedData.segment_jf_map.get(str(seg), [np.nan] * 24)[self.tod[0] : self.tod[-1]] )
                        jf_values_list.append(jf_values)
                    segment_jf_values.append(jf_values_list)

                bbox_segment_df['segment_jf_values'] = segment_jf_values # one value for each segment, multiple values for each bbox

            # ensure that each segment is associated with the same number of time steps
            a = []
            for key in self.SpeedData.segment_jf_map:
                a.append(len(self.SpeedData.segment_jf_map[key]))
            assert len(set(a)) == 1

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
                agg_func = lambda values, lengths: np.nansum(np.multiply(values, lengths)) / np.nansum(lengths)
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
