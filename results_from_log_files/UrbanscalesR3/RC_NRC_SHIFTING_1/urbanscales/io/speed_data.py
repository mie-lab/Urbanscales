import glob
import os
import pickle
import sys
import time

import numpy as np

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
    A class for handling and processing speed data for a specific city. It manages data loading,
    road segment mapping, and speed data aggregation for different temporal granularities.

    Attributes:
        city_name (str): The name of the city for which the speed data is processed.
        time_gran_minutes_raw (int): The granularity, in minutes, of the raw speed data.
        time_gran_minutes_target (int): The target granularity, in minutes, for aggregated speed data.
        road_segments (list): A list of road segments represented by line strings.
        NIDs (list): Node IDs corresponding to the road segments.
        NID_road_segment_map (dict): A mapping from node IDs to their corresponding road segments.
        nid_jf_map (dict): A mapping from node IDs to their jam factor (jf) values across time.
        segment_jf_map (dict): A mapping from road segments to their jam factor (jf) values.

    Methods:
        __init__(city_name, time_gran_minutes_raw, time_gran_minutes_target): Initializes the SpeedData object.
        set_road_segments(): Processes and sets the road segments from configuration data.
        set_segment_jf_map(): Processes and sets the jam factor mapping for road segments.
        _aggregation_mean(jf_list, combine_how_many_t_steps): Helper method for aggregating jam factor data.

    Example:
        >>> sd = SpeedData('New York', 5, 60)
    """
    def __init__(self, city_name, time_gran_minutes_raw, time_gran_minutes_target):
        """
        Initializes the SpeedData object with specified city and granularity of time for raw and target speed data.

        Parameters:
            city_name (str): The city for which to process the speed data.
            time_gran_minutes_raw (int): The granularity, in minutes, of the raw speed data.
            time_gran_minutes_target (int): The granularity, in minutes, targeted for aggregated speed data processing.
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
        """
        Loads and processes road segments from a specified data source. This method is configured to handle data path
        issues and set up necessary directories if missing.
        """
        # this chdir might not be needed;
        # there was some trouble with paths in my case.
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
        """
        Processes and maps jam factor data to corresponding road segments based on node IDs and timestamps. This method
        handles the temporal aggregation of jam factors to match the target time granularity.
        """
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
            elif config.CONGESTION_TYPE == "NON-RECURRENT": # legacy condition; we do not use this in the paper. The
                # metric being used in the paper is based on the NRCI index defined in SI, which corresponds to the
                # next condition: "NON-RECURRENT-MMM"
                self.segment_jf_map[Segment.seg_hash(self.NID_road_segment_map[seg_nid])] = copy.deepcopy(
                    (np.nanmax(np.array(a), axis=0) - np.nanmedian(np.array(a), axis=0)).tolist())
            elif config.CONGESTION_TYPE == "NON-RECURRENT-MMM":
                daily_differences = np.array(a) - np.nanmedian(np.array(a), axis=0)
                # Retain only positive differences, set negative differences to 0
                daily_differences[daily_differences < 0] = 0
                # Calculate the mean of these positive differences
                mean_positive_differences = np.nanmean(daily_differences, axis=0).tolist()
                self.segment_jf_map[Segment.seg_hash(self.NID_road_segment_map[seg_nid])] = copy.deepcopy(
                    mean_positive_differences)

        fname = os.path.join(config.BASE_FOLDER, config.network_folder, self.city_name, "_speed_data_object.pkl")


        if 2==2 or config.MASTER_VISUALISE_EACH_STEP:

            plt.clf()

            segment_counter = 0
            hist = []
            for segment_str, values in tqdm(self.segment_jf_map.items(), desc="Plotting the linestrings for "
                                                                                  "visualisation"):
                segment_counter += 1

                # Use Shapely to parse the MULTILINESTRING
                multiline = loads(segment_str)

                # Normalize mean_value for color mapping, adjust as per your color scale
                hist.append(np.mean(values))


            # Finalizing and showing the plot
            plt.hist(hist, bins=10)
            if not os.path.exists(os.path.join(config.network_folder, self.city_name, "raw_speed_data")):
                os.mkdir(os.path.join(config.network_folder, self.city_name, "raw_speed_data"))
            plt.savefig(os.path.join(config.network_folder, self.city_name, "raw_speed_data",
                                     "histogram" + ".png"), dpi=600)


            from matplotlib.cm import ScalarMappable

            fig, ax = plt.subplots()

            # Define the color map and normalization
            cmap = plt.cm.gist_rainbow
            norm = plt.Normalize(vmin=0, vmax=1)  # Adjust vmin and vmax based on your data range

            segment_counter = 0
            for segment_str, values in tqdm(self.segment_jf_map.items(),
                                            desc="Plotting the linestrings for visualisation"):
                segment_counter += 1

                # Use Shapely to parse the MULTILINESTRING
                multiline = loads(segment_str)

                # Normalize mean_value for color mapping, adjust as per your color scale
                color = cmap(norm(np.mean(values) * 0.99))  # * 0.99 to ensure always less than 1

                # Iterate through each line string in the multi-line string
                for linestring in multiline:
                    x, y = linestring.xy
                    ax.plot(x, y, color=color, linewidth=0.3)

            # Finalizing and showing the plot
            ax.set_aspect('equal', adjustable='datalim')

            # Create a ScalarMappable and use it to create the colorbar
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # You can set an empty array since the colorbar is based on the cmap and norm
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Mean Values')  # Set the label for the colorbar

            # Save the figure
            if not os.path.exists(os.path.join(config.network_folder, self.city_name, "raw_speed_data")):
                os.mkdir(os.path.join(config.network_folder, self.city_name, "raw_speed_data"))
            plt.savefig(os.path.join(config.network_folder, self.city_name, "raw_speed_data",
                                     "visualise_segment_boundaries.png"), dpi=600)

        if 1 == 2 or config.MASTER_VISUALISE_EACH_STEP:  # same thing at different TOD
            for tod in range(24):
                fig, ax = plt.subplots()

                segment_counter = 0
                for segment_str, values in tqdm(self.segment_jf_map.items(), desc="Plotting the linestrings for "
                                                                                      "visualisation"):
                    segment_counter += 1

                    # Use Shapely to parse the MULTILINESTRING
                    multiline = loads(segment_str)

                    # Normalize mean_value for color mapping, adjust as per your color scale
                    color = plt.cm.gist_rainbow(values[tod]/10 * 0.99)  # * 0.99 to ensure always less than 1

                    # Iterate through each line string in the multi-line string
                    for linestring in multiline:
                        x, y = linestring.xy
                        ax.plot(x, y, color=color, linewidth=0.4)

                # Finalizing and showing the plot
                ax.set_aspect('equal', adjustable='datalim')
                if not os.path.exists(os.path.join(config.network_folder, self.city_name, "raw_speed_data")):
                    os.mkdir(os.path.join(config.network_folder, self.city_name, "raw_speed_data"))
                plt.savefig(os.path.join(config.network_folder, self.city_name, "raw_speed_data",
                                         "visualise_segment_boundaries_with_tod_" +str(tod)+ ".png"), dpi=600)



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



    def _aggregation_mean(self, jf_list, combine_how_many_t_steps):
        """
        Aggregates jam factor data into a specified time granularity by averaging values.

        Parameters:
            jf_list (list): A list of jam factor values.
            combine_how_many_t_steps (int): Number of raw time steps to combine into a single target time step.

        Returns:
            list: A list of aggregated jam factor values.
        """

        agg_func = np.nanmean

        a = []
        for i in range(0, len(jf_list), combine_how_many_t_steps):
            a.append(agg_func(jf_list[i : i + combine_how_many_t_steps]))
        return a

    @staticmethod
    def organise_files_into_folders_all_cities(root_path):
        """
        Organizes files into designated folders for all cities. This method assumes that city-specific data files
        are scattered in a common directory and need to be moved into organized, city-specific subdirectories.

        Parameters:
            root_path (str): The root directory path where city-specific folders will be created and files moved into.

        This method moves each city's relevant files from a common directory to a dedicated subdirectory for that city,
        creating a structured data repository. It specifically handles 'linestring' and 'jam factor' (jf) files,
        sorting them into the appropriate city folders as configured.
        """
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
        """
        Processes and organizes speed data files for all cities specified in the configuration. This includes file
        sorting, directory organization, and initial processing of speed data objects for each city.
        """
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
    """
    Represents a road segment as a linestring. This class encapsulates the functionality for managing and manipulating
    geographical line data, specifically linestrings representing road segments.

    Attributes:
        line_string (shapely.geometry.LineString): A Shapely LineString object derived from the provided WKT linestring.
    """

    def __init__(self, linestring):
        """
        Initializes a Segment object from a Well-Known Text (WKT) linestring representation.

        Parameters:
            linestring (str): A WKT representation of a linestring, typically representing a road segment.
                              Format example: "MULTILINESTRING ((103.81404 1.32806 0, 103.81401 ...))"
        """
        self.line_string = shapely.wkt.loads(linestring)

    def get_shapely_linestring(self):
        """
        Returns the Shapely LineString object associated with this segment.

        Returns:
            shapely.geometry.LineString: The Shapely object representing the linestring.
        """
        return self.line_string

    def seg_hash(shapely_poly):
        """
        Generates a hash representation for a given Shapely polygon or linestring.

        Parameters:
            shapely_poly (shapely.geometry.Polygon or shapely.geometry.MultiLineString or Segment): The Shapely geometry
            object or another Segment instance for which a hash (WKT string) is desired.

        Returns:
            str: The Well-Known Text (WKT) string of the geometry, which serves as a hash.
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
    """
    A collection of Segment objects. This class is used to manage and manipulate lists of road segments represented
    as linestrings.

    Attributes:
        list_of_linestrings (list of Segment): A list of Segment objects representing road segments.
    """
    def __init__(self, list_of_linestrings):
        """
        Initializes the SegmentList with a list of linestrings, each represented as a Segment object.

        Parameters:
            list_of_linestrings (list of str): A list of Well-Known Text (WKT) linestring representations.
        """
        self.list_of_linestrings = []
        for ls in list_of_linestrings:
            self.list_of_linestrings.append(Segment(ls))


if __name__ == "__main__":
    SpeedData.preprocess_speed_data_for_all_cities()
