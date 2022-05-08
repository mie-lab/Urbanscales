import datetime
import math
import os
import json
import csv
import fnmatch
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import time

GEO_DISTANCE = True


def get_incident_from_json(json_file, incident_file):
    """

    Args:
        json_file: incident file from Here
        incident_file: grade, type, start_time, end_time, o_lat, o_lon, d_lat, d_lon

    Returns:

    """
    # csv file
    with open(incident_file, "w", newline="") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["grade", "type", "start_time", "end_time", "o_lat", "o_lon", "d_lat", "d_lon"])

        # json file
        with open(json_file) as f_json:
            d1 = json.load(f_json)
            d2 = d1["TRAFFICITEMS"]["TRAFFICITEM"]
            for i in range(len(d2)):
                item = d2[i]

                item_list = [item["CRITICALITY"]["DESCRIPTION"]]
                item_list.append(item["TRAFFICITEMTYPEDESC"])
                item_list.append(item["STARTTIME"])
                item_list.append(item["ENDTIME"])

                o_loc = item["LOCATION"]["GEOLOC"]["ORIGIN"]
                item_list.append(o_loc["LATITUDE"])
                item_list.append(o_loc["LONGITUDE"])

                d_loc = item["LOCATION"]["GEOLOC"]["TO"]
                for j in range(len(d_loc)):
                    _d_loc = d_loc[j]
                    item_list.append(_d_loc["LATITUDE"])
                    item_list.append(_d_loc["LONGITUDE"])

                csv_writer.writerow(item_list)


def get_congestion_from_json_perday(json_folder, incident_csv, incident_fig, date, n_gap):
    """

    Args:
        json_folder: folder for incident files from Here
        incident_csv: per day, grade, type, start_time, end_time, o_lat, o_lon, d_lat, d_lon, dist
        incident_fig: per day, frequency of distance
        date: used for regular expression, e.g., '2022-02-21' -> 'incident*2022-02-21*minor.json'
        n_gap: calculate every n_gap file

    Returns:

    """
    json_list = []
    print(json_folder)
    for root, dirs, files in os.walk(json_folder):
        json_list.append(files)
    # print(json_list)
    json_list = json_list[0]

    minor_list = []
    re_match = "incident*" + date + "*minor.json"
    # r"..\incident-2022-02-21-18_58_47_minor.json"
    for i_file in json_list:
        if fnmatch.fnmatch(i_file, re_match):
            minor_list.append(i_file)

    minor_list.sort()
    print("file number:", len(minor_list))

    # csv file
    with open(incident_csv, "w", newline="") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(
            ["originID", "grade", "type", "start_time", "end_time", "o_lat", "o_lon", "d_lat", "d_lon", "dist"]
        )

        all_distance = []
        # for json_file in minor_list:
        for j_file in range(len(minor_list)):
            # json file
            if j_file % n_gap != 0:  # extraction every n_gap file
                continue
            json_file = minor_list[j_file]
            json_file = json_folder + "/" + json_file
            with open(json_file) as f_json:
                d1 = json.load(f_json)
                d2 = d1["TRAFFICITEMS"]["TRAFFICITEM"]
                for i in range(len(d2)):
                    item = d2[i]

                    # Only congestion is considered
                    if item["TRAFFICITEMTYPEDESC"] != "CONGESTION":
                        continue

                    origin_ID = str(item["ORIGINALTRAFFICITEMID"])

                    item_list = [origin_ID]
                    item_list.append(item["CRITICALITY"]["DESCRIPTION"])
                    item_list.append(item["TRAFFICITEMTYPEDESC"])
                    item_list.append(item["STARTTIME"])
                    item_list.append(item["ENDTIME"])

                    o_loc = item["LOCATION"]["GEOLOC"]["ORIGIN"]
                    item_list.append(o_loc["LATITUDE"])
                    item_list.append(o_loc["LONGITUDE"])
                    coords_1 = (float(o_loc["LATITUDE"]), float(o_loc["LONGITUDE"]))

                    d_loc = item["LOCATION"]["GEOLOC"]["TO"]
                    for j in range(len(d_loc)):
                        _d_loc = d_loc[j]
                        item_list.append(_d_loc["LATITUDE"])
                        item_list.append(_d_loc["LONGITUDE"])
                        coords_2 = (float(_d_loc["LATITUDE"]), float(_d_loc["LONGITUDE"]))

                        _dist = 0.0
                        if GEO_DISTANCE:  # Geodesic distance
                            _dist = geodesic(coords_1, coords_2).km
                        else:  # Euclidean distance
                            _dist = math.dist(coords_1, coords_2)
                        all_distance.append(_dist)
                        item_list.append(_dist)

                    item_list[0] = "'" + str(item_list[0]) + "'"
                    csv_writer.writerow(item_list)

        # visualization
        plt.hist(all_distance, bins=40, facecolor="blue", edgecolor="black")
        plt.xlabel("distance")
        plt.ylabel("frequency")
        plt.title("Distance frequency in %s" % (date))
        plt.savefig(incident_fig)
        plt.close()


def get_congestion_from_json_perday_uniqueID(json_folder, incident_csv, incident_fig, date):
    """
    We use a unique ID (ORIGINALTRAFFICITEMID) to record incidents.
    The start time and origin coordinate are from the first file of ORIGINALTRAFFICITEMID.
    The end time and destination coordinate are from the last file of ORIGINALTRAFFICITEMID.

    Args:
        json_folder: folder for incident files from Here
        incident_csv: per day - originID, grade, type, start_time, end_time, lasting_time, o_lat, o_lon, d_lat, d_lon, dist
        incident_fig: per day, frequency of distance
        date: used for regular expression, e.g., '2022-02-21' -> 'incident*2022-02-21*minor.json'

    Returns:

    """
    json_list = []
    print(json_folder)
    for root, dirs, files in os.walk(json_folder):
        json_list.append(files)
    # print(json_list)
    json_list = json_list[0]

    minor_list = []
    re_match = "incident*" + date + "*minor.json"
    # r"..\incident-2022-02-21-18_58_47_minor.json"
    for i_file in json_list:
        if fnmatch.fnmatch(i_file, re_match):
            minor_list.append(i_file)

    minor_list.sort()
    print("file number:", len(minor_list))

    # csv file
    with open(incident_csv, "w", newline="") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(
            [
                "originID",
                "grade",
                "type",
                "start_time",
                "end_time",
                "lasting_time",
                "o_lat",
                "o_lon",
                "d_lat",
                "d_lon",
                "dist (km)",
            ]
        )

        dict_incident = {}  # record and update the time and locations
        dict_distance = {}  # record distance for visulization

        for j_file in range(len(minor_list)):
            # json file
            json_file = minor_list[j_file]
            json_file = json_folder + "/" + json_file
            try:
                with open(json_file) as f_json:

                    d1 = json.load(f_json)
                    d2 = d1["TRAFFICITEMS"]["TRAFFICITEM"]
                    for i in range(len(d2)):  # incident items
                        item = d2[i]
                        if item["TRAFFICITEMTYPEDESC"] != "CONGESTION":  # only consider congestion
                            continue

                        origin_ID = str(item["ORIGINALTRAFFICITEMID"])

                        if origin_ID in dict_incident:
                            # update the end_time, destination_coord and dist
                            dict_incident[origin_ID][4] = item["ENDTIME"]
                            lasting_time = get_time_difference(dict_incident[origin_ID][3], dict_incident[origin_ID][4])
                            dict_incident[origin_ID][5] = lasting_time

                            coords_1 = (float(dict_incident[origin_ID][6]), float(dict_incident[origin_ID][7]))

                            d_loc = item["LOCATION"]["GEOLOC"]["TO"]
                            for j in range(1):
                                _d_loc = d_loc[j]
                                dict_incident[origin_ID][8] = _d_loc["LATITUDE"]
                                dict_incident[origin_ID][9] = _d_loc["LONGITUDE"]
                                coords_2 = (float(_d_loc["LATITUDE"]), float(_d_loc["LONGITUDE"]))

                                _dist = 0.0
                                if GEO_DISTANCE:  # Geodesic distance
                                    _dist = geodesic(coords_1, coords_2).km
                                else:  # Euclidean distance
                                    _dist = math.dist(coords_1, coords_2)
                                dict_incident[origin_ID][10] = _dist
                                dict_distance[origin_ID] = _dist
                        else:
                            # Only congestion is considered
                            item_list = [origin_ID]
                            item_list.append(item["CRITICALITY"]["DESCRIPTION"])
                            item_list.append(item["TRAFFICITEMTYPEDESC"])
                            item_list.append(item["STARTTIME"])
                            item_list.append(item["ENDTIME"])

                            lasting_time = get_time_difference(item["STARTTIME"], item["ENDTIME"])
                            item_list.append(lasting_time)

                            o_loc = item["LOCATION"]["GEOLOC"]["ORIGIN"]
                            item_list.append(o_loc["LATITUDE"])
                            item_list.append(o_loc["LONGITUDE"])
                            coords_1 = (float(o_loc["LATITUDE"]), float(o_loc["LONGITUDE"]))

                            d_loc = item["LOCATION"]["GEOLOC"]["TO"]
                            for j in range(1):
                                _d_loc = d_loc[j]
                                item_list.append(_d_loc["LATITUDE"])
                                item_list.append(_d_loc["LONGITUDE"])
                                coords_2 = (float(_d_loc["LATITUDE"]), float(_d_loc["LONGITUDE"]))

                                _dist = 0.0
                                if GEO_DISTANCE:  # Geodesic distance
                                    _dist = geodesic(coords_1, coords_2).km
                                else:  # Euclidean distance
                                    _dist = math.dist(coords_1, coords_2)
                                item_list.append(_dist)
                                dict_distance[origin_ID] = _dist

                            dict_incident[origin_ID] = item_list
            except:
                print("File where crashed: ", json_file)
                continue
        # write dictionary
        for record in dict_incident.values():
            record[0] = "'" + str(record[0]) + "'"
            csv_writer.writerow(record)

        # visualization
        plt.hist(list(dict_distance.values()), bins=40, facecolor="blue", edgecolor="black")
        plt.xlabel("distance")
        plt.ylabel("frequency")
        plt.title("Distance frequency in %s" % (date))
        plt.savefig(incident_fig)
        plt.close()


def get_date_list(start, end):
    date_list = []
    date = datetime.datetime.strptime(start, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")
    while date <= end:
        date_list.append(date.strftime("%Y-%m-%d"))
        date = date + datetime.timedelta(1)
    return date_list


def get_time_difference(start, end):
    start = datetime.datetime.strptime(start, "%m/%d/%Y %H:%M:%S")
    end = datetime.datetime.strptime(end, "%m/%d/%Y %H:%M:%S")
    delta_time = end - start
    return delta_time


if __name__ == "__main__":
    # get_incident_from_json(r"..\incident-2022-02-21-18_58_47_minor.json", r"..\test.csv")
    n_gap = 1  # extraction every n_gap file
    date_list = get_date_list("2022-02-21", "2022-03-06")

    for date in date_list:
        print("date: ", date)
        # print("file gap: ", n_gap)
        # get_congestion_from_json_perday(r"/usr1/data_share_here/incidents_folder",
        #                                 "incident_%s_stats_every%d.csv" % (date, n_gap),
        #                                 "incident_%s_stats_every%d.png" % (date, n_gap),
        #                                 date,
        #                                 n_gap)
        get_congestion_from_json_perday_uniqueID(
            r"/usr1/data_share_here/incidents_folder",
            "incident_%s_stats_uniqueID.csv" % (date),
            "incident_%s_stats_uniqueID.png" % (date),
            date,
        )
