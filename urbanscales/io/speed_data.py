import glob
import pickle
import sys
import time

import numpy as np

import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import config
import os
import shapely.wkt
from shapely import geometry
import pandas as pd
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from smartprint import smartprint as sprint
import shutil
from shapely.wkt import loads


class SpeedData:
    """
    self.city_name = city_name
    self.time_gran_minutes_raw = time_gran_minutes_raw
    self.time_gran_minutes_target = time_gran_minutes_target

    self.road_segments = None
    self.NIDs = None
    self.NID_road_segment_map = {}
    self.nid_jf_map = {}
    self.segment_jf_map = {}
    """

    def __init__(self, city_name, time_gran_minutes_raw, time_gran_minutes_target):
        """

        Args:
            city_name:
            time_gran_minutes_raw:
            time_gran_minutes_target:
        """

        fname = os.path.join(config.BASE_FOLDER, config.network_folder, city_name, "_speed_data_object.pkl")
        if config.sd_delete_existing_pickle_objects:
            if os.path.exists(fname):
                os.remove(fname)

        if os.path.exists(fname):
            with open(fname, "rb") as f:
                temp = copy.deepcopy(pickle.load(f))
                self.__dict__.update(temp.__dict__)
                print("Read speed data object from pickle")
        else:
            self.city_name = city_name
            self.time_gran_minutes_raw = time_gran_minutes_raw
            self.time_gran_minutes_target = time_gran_minutes_target

            self.road_segments = None
            self.NIDs = None
            self.NID_road_segment_map = {}
            self.nid_jf_map = {}
            self.segment_jf_map = {}

            self.set_road_segments()
            self.set_segment_jf_map()

    def set_road_segments(self):
        # this chdir might not be needed;
        # tgere was some trouble with paths in my case.
        os.chdir(config.home_folder_path)

        if not os.path.exists(os.path.join(config.sd_base_folder_path, self.city_name)):
            if not os.path.exists(config.sd_base_folder_path):
                os.mkdir(config.sd_base_folder_path)
            os.mkdir(os.path.join(config.sd_base_folder_path, self.city_name))

        if not os.path.exists(
            os.path.join(config.sd_base_folder_path, self.city_name, config.sd_seg_file_path_within_city)
        ):
            sprint(os.path.join(config.sd_base_folder_path, self.city_name, config.sd_seg_file_path_within_city))
            sprint(self.city_name, "Missing here data")
            raise Exception("Error in here data; data file SEG missing")
            # sys.exit(0)

        df = pd.read_csv(os.path.join(config.sd_base_folder_path, self.city_name, config.sd_seg_file_path_within_city))
        df = df[["NID", "Linestring"]].copy()
        self.road_segments = SegmentList(df.Linestring.to_list()).list_of_linestrings
        self.NIDs = df.NID.to_list()
        self.NID_road_segment_map = dict(zip(self.NIDs, self.road_segments))
        debug_stop = 2

    def set_segment_jf_map(self):
        if not os.path.exists(
            os.path.join(config.sd_base_folder_path, self.city_name, config.sd_jf_file_path_within_city)
        ):
            raise Exception("Error in here data; data file JF missing")
        """
        NID	2022-02-28T23:59:27	2022-03-01T00:01:27	2022-03-01T00:03:27	2022-03-01T00:05:27
        0	0.0	0.18115	0.18115	0.18115
        1	0.0	0.0	0.0	0.0
        2	0.0	0.0	0.0	0.0
        3	0.0	0.0	0.0	0.0
        4	1.47707	1.54911	1.20348	0.99297
        """
        df = pd.read_csv(os.path.join(config.sd_base_folder_path, self.city_name, config.sd_jf_file_path_within_city))

        # column strings converted to datetime headers for comparing with range
        datetime_header = pd.to_datetime(df.columns[1:]).tz_convert(config.rn_city_wise_tz_code[self.city_name])

        assert self.road_segments is not None, "list_of_linestrings not set"

        self.num_timesteps_in_data = 0

        for i in tqdm(range(len(self.NIDs)), desc=" Reading JF file"):
            seg_nid = self.NIDs[i]

            jf_list = (
                (
                    df.loc[df["NID"] == seg_nid][
                        df.columns[1:][
                            (
                                datetime_header
                                >= pd.to_datetime(config.sd_start_datetime_str).tz_localize(
                                    config.rn_city_wise_tz_code[self.city_name]
                                )
                            )
                            & (
                                datetime_header
                                <= pd.to_datetime(config.sd_end_datetime_str).tz_localize(
                                    config.rn_city_wise_tz_code[self.city_name]
                                )
                            )
                        ]
                    ]
                )
                .values.flatten()
                .tolist()
            )

            # 10 minutes to 1 hour frozen to mean
            jf_list = self._aggregation_mean(jf_list, self.time_gran_minutes_target // self.time_gran_minutes_raw)

            if self.num_timesteps_in_data == 0:
                self.num_timesteps_in_data = len(jf_list)
            else:
                # all segments should have the same number of time steps in data
                assert self.num_timesteps_in_data == len(jf_list)

            self.nid_jf_map[seg_nid] = copy.deepcopy(jf_list)


            a = []
            plt.clf()
            for i in range(30):
                if config.MASTER_VISUALISE_EACH_STEP:
                    plt.plot(jf_list[i * 24: (i + 1) * 24], alpha=0.2, color="blue")

                if len(jf_list[i * 24: (i + 1) * 24]) == 24:
                    a.append(jf_list[i * 24: (i + 1) * 24]) # so that we can extract the mean day
                    # a.append(jf_list[i * 24 + 6 : i * 24 + 9])


            if config.MASTER_VISUALISE_EACH_STEP:
                plt.plot(np.nanmean(np.array(a), axis=0), linewidth=4, color="black", label="mean_tod_plot")
                plt.fill_between(range(len(a[0])), [0] * 24, np.nanmean(np.array(a), axis=0), color="gray", alpha=0.3)

                plt.legend()
                plt.ylim(0, 10)
                if not os.path.exists(os.path.join(config.network_folder, self.city_name, "mean_day")):
                    os.mkdir(os.path.join(config.network_folder, self.city_name, "mean_day"))
                r = str(int(np.random.rand() * 100000000))
                plt.xlabel("Hour of day")
                plt.savefig(os.path.join(config.network_folder, self.city_name, "mean_day",
                                         f"mean_day_{r}" + ".png"), dpi=300)
                # plt.show(block=False)
                plt.clf()
                plt.plot(np.nanmedian(np.array(a), axis=0), linewidth=4, color="blue", label="median_day")
                plt.plot(np.nanmax(np.array(a), axis=0), linewidth=4, color="black", label="max_day")
                plt.fill_between(range(len(a[0])), np.nanmedian(np.array(a), axis=0), np.nanmax(np.array(a), axis=0),
                                 color="gray", alpha=0.3)
                plt.legend()
                plt.xlabel("Hour of day")
                plt.ylim(0, 10)
                if not os.path.exists(os.path.join(config.network_folder, self.city_name, "mean_day")):
                    os.mkdir(os.path.join(config.network_folder, self.city_name, "mean_day"))
                r = str(int(np.random.rand() * 100000000))
                plt.savefig(os.path.join(config.network_folder, self.city_name, "mean_day",
                                         f"_max_minus_median_day_{r}" + ".png"), dpi=300)
                # plt.show(block=False)

            # self.segment_jf_map[Segment.seg_hash(self.NID_road_segment_map[seg_nid])] = copy.deepcopy(jf_list)
            if config.CONGESTION_TYPE == "RECURRENT":
                self.segment_jf_map[Segment.seg_hash(self.NID_road_segment_map[seg_nid])] = copy.deepcopy(
                    np.nanmean(np.array(a), axis=0).tolist())
            elif config.CONGESTION_TYPE == "NON-RECURRENT":
                self.segment_jf_map[Segment.seg_hash(self.NID_road_segment_map[seg_nid])] = copy.deepcopy(
                    (np.nanmax(np.array(a), axis=0) - np.nanmedian(np.array(a), axis=0)).tolist())

        fname = os.path.join(config.BASE_FOLDER, config.network_folder, self.city_name, "_speed_data_object.pkl")


        if config.MASTER_VISUALISE_EACH_STEP:


            fig, ax = plt.subplots()

            segment_counter = 0
            for segment_str, values in tqdm(self.segment_jf_map.items(), desc="Plotting the linestrings for "
                                                                                  "visualisation"):
                segment_counter += 1

                # Use Shapely to parse the MULTILINESTRING
                multiline = loads(segment_str)

                # Normalize mean_value for color mapping, adjust as per your color scale
                color = plt.cm.gist_rainbow(np.random.rand() * 0.99)  # * 0.99 to ensure always less than 1

                # Iterate through each line string in the multi-line string
                for linestring in multiline:
                    x, y = linestring.xy
                    ax.plot(x, y, color=color, linewidth=0.4)

            # Finalizing and showing the plot
            ax.set_aspect('equal', adjustable='datalim')
            if not os.path.exists(os.path.join(config.network_folder, self.city_name, "raw_speed_data")):
                os.mkdir(os.path.join(config.network_folder, self.city_name, "raw_speed_data"))
            plt.savefig(os.path.join(config.network_folder, self.city_name, "raw_speed_data",
                                     "visualise_segment_boundaries" + ".png"), dpi=600)

            # Normalize color scale by the max mean of the values for dynamic color range
            if config.MASTER_VISUALISE_EACH_STEP:
                for tod_counter in [0] :# range(0, 1000):
                    fig, ax = plt.subplots()

                    max_mean_value = -1
                    for segment_str, values in tqdm(self.segment_jf_map.items(), desc="Plotting the linestrings for "
                                                                                      "visualisation"):

                        # Calculate mean of values for color mapping
                        # mean_value = np.nanmean(values)
                        # max_mean_value = max(max_mean_value, values[tod_counter])
                        max_mean_value = max(max_mean_value, self.segment_jf_map[segment_str][0])


                    for segment_str, values in tqdm(self.segment_jf_map.items(), desc="Plotting the linestrings for "
                                                                                      "visualisation"):
                        # Use Shapely to parse the MULTILINESTRING
                        multiline = loads(segment_str)

                        # Normalize mean_value for color mapping, adjust as per your color scale
                        # color = plt.cm.gist_rainbow(values[tod_counter] / max_mean_value * 0.99)  # Example color mapping
                        color = plt.cm.gist_rainbow(self.segment_jf_map[segment_str][0] / max_mean_value * 0.99)  # Example color mapping

                        # Iterate through each line string in the multi-line string
                        for linestring in multiline:
                            x, y = linestring.xy
                            ax.plot(x, y, color=color, linewidth=0.1)

                    # Finalizing and showing the plot
                    ax.set_aspect('equal', adjustable='datalim')
                    plt.savefig(os.path.join(config.network_folder, self.city_name, "raw_speed_data", "plot" + str(tod_counter)
                                             + ".png"), dpi=1800)
                print("Plot updated")

        avg_ = []
        for i in range(len(jf_list)//(1440//config.sd_target_speed_data_gran)):
            avg_.append(jf_list[i * 24:(i + 1) * 24])

        avg_ = np.array(avg_)
        plt.plot(np.mean(avg_, axis=0))
        plt.title(self.city_name)
        plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, self.city_name, "_speed_plot_raw_data_aggregated.png"))

        # make the folder if it dfoes not exist
        if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, self.city_name)):
            os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, self.city_name))


        if not os.path.exists(fname):
            rand_pickle_marker = os.path.join(config.temp_folder_for_robust_pickle_files, str(int(np.random.rand() * 100000000000000)) )
            with open(rand_pickle_marker, "wb") as f:
                pickle.dump(self, f, protocol=config.pickle_protocol)
            os.rename(rand_pickle_marker, fname)

    # def get_object(cityname):
    #     """
    #
    #     Args:
    #         scale:
    #
    #     Returns: (Saved) Object of this class (Scale)
    #
    #     """
    #     fname = os.path.join("network", cityname, "_speed_data_object.pkl")
    #     if os.path.exists(fname):
    #         with open(fname, "rb") as f:
    #             # try:
    #             obj = pickle.load(f)
    #             return obj
    #             # except AttributeError:
    #             #     raise Exception("Something wrong with speed data object pickle, run again after deleting\
    #             #                     the pickle file _speed_data_object.pkl and run speed_data.py again")

    # try:
    #     return obj
    # except UnboundLocalError:
    #     print(
    #         "Error in get_object(), probably due to speed data not \
    #                    processed; existing execution"
    #     )
    #     sys.exit()


    def _aggregation_mean(self, jf_list, combine_how_many_t_steps):
        """

        Args:
            jf_list:
            combine_how_many_t_steps:

        Returns:
            shortened JF list

        """

        agg_func = np.nanmean

        a = []
        for i in range(0, len(jf_list), combine_how_many_t_steps):
            a.append(agg_func(jf_list[i : i + combine_how_many_t_steps]))
        return a

    @staticmethod
    def organise_files_into_folders_all_cities(root_path):
        for city in config.scl_master_list_of_cities:
            if not os.path.exists(os.path.join(config.BASE_FOLDER, root_path, city)):
                os.mkdir(os.path.join(root_path, city))

            for filename in glob.glob(os.path.join(root_path, "*")):
                if city in filename:
                    if "linestring" in filename:
                        shutil.move(
                            os.path.join(filename),
                            os.path.join(root_path, city, config.sd_seg_file_path_within_city),
                        )
                    if "jf" in filename:
                        shutil.move(
                            os.path.join(filename),
                            os.path.join(root_path, city, config.sd_jf_file_path_within_city),
                        )

    @staticmethod
    def preprocess_speed_data_for_all_cities():
        SpeedData.organise_files_into_folders_all_cities(config.sd_base_folder_path)
        for city in config.scl_master_list_of_cities:
            for seed in config.scl_list_of_seeds:
                for depth in config.scl_list_of_depths:
                    print(city, seed, depth)
                    startime = time.time()
                    sd = SpeedData(city, config.sd_raw_speed_data_gran, config.sd_target_speed_data_gran)
                    print(sd.num_timesteps_in_data)
                    print(time.time() - startime)


class Segment:
    def __init__(self, linestring):
        """

        Args:
            linestring: format of linestring
                        MULTILINESTRING ((103.81404 1.32806 0,103.81401 ....... 103.81363 1.32444 0,103.81353 1.32388 0
                        ,103.81344 1.32346 0))
        """
        self.line_string = shapely.wkt.loads(linestring)

    def get_shapely_linestring(self):
        return self.line_string

    def seg_hash(shapely_poly):
        """

        Args:
            shapely_poly: Shapely polygon
                    if it is in string (wkt), no need for this function, but no harm as well
                                            just make sure that it is the same throughout
                                            don't mix with and without hash

        Returns:
            hash

        """
        assert (
            isinstance(shapely_poly, str)
            or isinstance(shapely_poly, geometry.MultiLineString)
            or isinstance(shapely_poly, Segment)
        )
        if isinstance(shapely_poly, geometry.Polygon) or isinstance(shapely_poly, geometry.MultiLineString):
            return shapely_poly.wkt
        elif isinstance(shapely_poly, Segment):
            return shapely_poly.line_string.wkt


class SegmentList:
    def __init__(self, list_of_linestrings):
        """

        Args:
            list_of_linestrings:
        """
        self.list_of_linestrings = []
        for ls in list_of_linestrings:
            self.list_of_linestrings.append(Segment(ls))


if __name__ == "__main__":
    SpeedData.preprocess_speed_data_for_all_cities()
