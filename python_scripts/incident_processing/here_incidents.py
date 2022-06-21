import datetime
import math
import os
import json
import csv
import fnmatch
import collections
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import pandas as pd
import datetime, time, re
import holidays
import seaborn as sns, matplotlib.pyplot as plt, operator as op

GEO_DISTANCE = True


def get_date_list(start, end):
    """
    Generate a list of date from start_date to end_date

    Parameters
    ----------
    start : string with a format of %Y-%m-%d
        The start date of the date list.
    end : string with a format of %Y-%m-%d
        The end date of the date list.

    Returns
    -------
    date_list : list
        The date list.

    """
    date_list = []
    date = datetime.datetime.strptime(start, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")
    while date <= end:
        date_list.append(date.strftime("%Y-%m-%d"))
        date = date + datetime.timedelta(1)
    return date_list


def get_time_difference(start, end):
    """
    Compute the time difference between start_date and end_date, with format %H:%M:%S

    Parameters
    ----------
    start : string with a format of %m/%d/%Y %H:%M:%S
    end : string with a format of %m/%d/%Y %H:%M:%S

    Returns
    -------
    delta_time : datetime
        The time difference with a format of %H:%M:%S.

    """
    start = datetime.datetime.strptime(start, "%m/%d/%Y %H:%M:%S")
    end = datetime.datetime.strptime(end, "%m/%d/%Y %H:%M:%S")
    delta_time = end - start
    return delta_time


def get_rel_time_difference_seconds(time1, time2):
    """
    Compute the relative time difference between time1 and time2 by seconds

    Parameters
    ----------
    time1 : string with a format of %m/%d/%Y %H:%M:%S
    time2 : string with a format of %m/%d/%Y %H:%M:%S

    Returns
    -------
    Relative time difference by seconds

    """
    time1_list = re.split(r"[ .Z]", time1)
    time2_list = re.split(r"[ .Z]", time2)
    try:
        _time1 = datetime.datetime.strptime(time1_list[0], "%Y-%m-%dT%H:%M:%S")
        _time2 = datetime.datetime.strptime(time2_list[0], "%Y-%m-%dT%H:%M:%S")
        delta_time_1 = _time2 - _time1
        delta_time_2 = _time1 - _time2
        rel_time = min(abs(delta_time_1.total_seconds()), abs(delta_time_2.total_seconds()))
        return rel_time
    except:
        print(time1, time2)
        return "ERR"


# obtain sepcific file list from all files
def obtain_file_list(re_expression, in_dir):
    """
    Generate the list of files satisfying the re-expression

    Parameters
    ----------
    re_expression : re expression
    in_dir : file dir path

    Returns
    -------
    file_list : list, path + filename.
    file_name : list, only filename.
    file_base : list, only filebase.

    """
    _file_name = os.listdir(in_dir)
    file_name = []
    # print(_file_name)
    for i in _file_name:
        if re.match(re_expression, i):
            file_name.append(i)
    # print(file_name)
    file_name.sort()
    file_base = []
    file_list = []
    for item in file_name:
        file_list.append(in_dir + "/" + item)
        _base1 = item.split(".")
        _base1 = _base1[0:-1]
        _base2 = ".".join(_base1)
        file_base.append(_base2)
    return file_list, file_name, file_base


def utc2local(utc_time):
    """
    Convert from utc time to local time

    Parameters
    ----------
    utc_time : TYPE-datetime, utc time.

    Returns
    -------
    local_time : TYPE-datetime, local time.

    """
    now_timestamp = time.time()
    offset = datetime.datetime.fromtimestamp(now_timestamp) - datetime.datetime.utcfromtimestamp(now_timestamp)
    local_time = utc_time + offset
    return local_time


def isRestday(local_date):
    """
    Identify whether a date is a Restday or Workday in Singapore.
    Restday includes public holiday and weekends.

    Parameters
    ----------
    local_date : TYPE-datetime, local date of Singapore.

    Returns
    -------
    True: rest day
    False: work day

    """
    isHoliday = False
    isWeekend = False

    holiday_list = holidays.Singapore(years=[2021, 2022, 2023]).keys()
    if local_date in holiday_list:
        isHoliday = True
    if local_date.weekday() > 4:
        isWeekend = True
    return isHoliday or isWeekend


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
    The congestion_factor is from the first file of ORIGINALTRAFFICITEMID.

    Args:
        json_folder: folder for incident files from Here
        incident_csv: per day - originID, grade, type, congestion_factor, start_time, end_time, lasting_time, o_lat, o_lon, d_lat, d_lon, dist
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
                "congestion_factor",
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
                            dict_incident[origin_ID][3] = item["TRAFFICITEMDETAIL"]["INCIDENT"]["CONGESTIONINCIDENT"][
                                "CONGESTIONFACTOR"
                            ]
                            # update the end_time, destination_coord and dist
                            dict_incident[origin_ID][5] = item["ENDTIME"]
                            lasting_time = get_time_difference(dict_incident[origin_ID][4], dict_incident[origin_ID][5])
                            dict_incident[origin_ID][6] = lasting_time

                            coords_1 = (float(dict_incident[origin_ID][7]), float(dict_incident[origin_ID][8]))

                            d_loc = item["LOCATION"]["GEOLOC"]["TO"]
                            for j in range(1):
                                _d_loc = d_loc[j]
                                dict_incident[origin_ID][9] = _d_loc["LATITUDE"]
                                dict_incident[origin_ID][10] = _d_loc["LONGITUDE"]
                                coords_2 = (float(_d_loc["LATITUDE"]), float(_d_loc["LONGITUDE"]))

                                _dist = 0.0
                                if GEO_DISTANCE:  # Geodesic distance
                                    _dist = geodesic(coords_1, coords_2).km
                                else:  # Euclidean distance
                                    _dist = math.dist(coords_1, coords_2)
                                dict_incident[origin_ID][11] = _dist
                                dict_distance[origin_ID] = _dist
                        else:
                            # Only congestion is considered
                            item_list = [origin_ID]
                            item_list.append(item["CRITICALITY"]["DESCRIPTION"])
                            item_list.append(item["TRAFFICITEMTYPEDESC"])
                            # This congestion factor is only available for congestion
                            item_list.append(
                                item["TRAFFICITEMDETAIL"]["INCIDENT"]["CONGESTIONINCIDENT"]["CONGESTIONFACTOR"]
                            )
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


def sns_box_plot_from_dict(dict_data, fig_file):
    # sort keys and values together
    sorted_keys, sorted_vals = zip(*sorted(dict_data.items(), key=op.itemgetter(1)))

    # almost verbatim from question
    sns.utils.axlabel(xlabel="Groups", ylabel="Y-Axis", fontsize=16)
    sns.boxplot(data=sorted_vals, width=0.18)
    sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=0.9)

    # category labels
    plt.xticks(plt.xticks()[0], sorted_keys)
    plt.savefig(fig_file)


if __name__ == "__main__":
    print("test")

# =============================================================================
#     # export incident data from json file to csv
#     # get_incident_from_json(r"..\incident-2022-02-21-18_58_47_minor.json", r"..\test.csv")
#     n_gap = 1  # extraction every n_gap file
#     date_list = get_date_list('2022-02-21', '2022-04-06')
#
#     for date in date_list:
#         print("date: ", date)
#         print("file gap: ", n_gap)
#         get_congestion_from_json_perday(r"/usr1/data_share_here/incidents_folder",
#                                         "incident_%s_stats_every%d.csv" % (date, n_gap),
#                                         "incident_%s_stats_every%d.png" % (date, n_gap),
#                                         date,
#                                         n_gap)
#         get_congestion_from_json_perday_uniqueID(r"/usr1/data_share_here/incidents_folder",
#                                                  "incident_%s_stats_factor_uniqueID.csv" % (date),
#                                                  "incident_%s_stats_factor_uniqueID.png" % (date),
#                                                  date)
#     print("file gap: ", n_gap)
#     get_congestion_from_json_perday_uniqueID(r"/usr1/yatao/here_data/here_data_20220406/",
#                                              "Incident_20220221_20220406/incident_%s_stats_factor_uniqueID.csv" % (date),
#                                              "Incident_20220221_20220406/incident_%s_stats_factor_uniqueID.png" % (date),
#                                              date)
# =============================================================================

# =============================================================================
#     # statistics of incidents time across hours in workday and restday separately
#     dict_rest_lasting_time = {}
#     dict_work_lasting_time = {}
#     file_list, file_name, file_base = obtain_file_list('([\s\S]*\.csv)', './here_incidents_data/incident_20220221_20220406')
#     for _file in file_list:
#         df = pd.read_csv(_file)
#         for index, row in df.iterrows():
#             _start_time = row['start_time']
#             _last_time = row['lasting_time']
#             _start_time1 = utc2local(datetime.datetime.strptime(_start_time, '%m/%d/%Y %H:%M:%S'))
#             _start_date = _start_time1.date()
#             _last_time1 = datetime.datetime.strptime(_last_time, '%H:%M:%S')
#
#             _hour = _start_time1.hour
#             _total_second = _last_time1.hour * 3600 + _last_time1.minute * 60 + _last_time1.second
#
#             _isRestday = isRestday(_start_date)
#             if _isRestday:
#                 if _hour in dict_rest_lasting_time:
#                     dict_rest_lasting_time[_hour].append(_total_second)
#                 else:
#                     dict_rest_lasting_time[_hour] = [_total_second]
#             else:
#                 if _hour in dict_work_lasting_time:
#                     dict_work_lasting_time[_hour].append(_total_second)
#                 else:
#                     dict_work_lasting_time[_hour] = [_total_second]
#
#     # csv file
#     rest_data_time = pd.DataFrame.from_dict(dict_rest_lasting_time, orient='index', columns=None)
#     rest_data_time.to_csv('./here_incidents_data/Restday_HourID_incidentLastingTime.csv', sep=',', index=True, index_label='hour_id')
#     work_data_time = pd.DataFrame.from_dict(dict_work_lasting_time, orient='index', columns=None)
#     work_data_time.to_csv('./here_incidents_data/Workday_HourID_incidentLastingTime.csv', sep=',', index=True, index_label='hour_id')
#
#
#     sort_dict_rest_lasting_time = collections.OrderedDict(sorted(dict_rest_lasting_time.items()))
#     sort_dict_work_lasting_time = collections.OrderedDict(sorted(dict_work_lasting_time.items()))
#     # sns_box_plot_from_dict(sort_dict_rest_lasting_time, './here_incidents_data/rest_fig.png')
# =============================================================================

# =============================================================================
#     # visulization by using box plots
#     labels, data = sort_dict_work_lasting_time.keys(), sort_dict_work_lasting_time.values()
#     plt.figure(dpi=300, figsize=(20,13))
#     plt.boxplot(data)
#     plt.xticks(range(1, len(labels) + 1), labels)
#     plt.xlabel('hour')
#     plt.ylabel('incident time')
#     plt.title('Incident time across hours in workday')
#     plt.savefig('./here_incidents_data/incident_time_workday_box_plot.png')
#     labels, data = sort_dict_rest_lasting_time.keys(), sort_dict_rest_lasting_time.values()
#     plt.figure(dpi=300, figsize=(20,13))
#     plt.boxplot(data)
#     plt.xticks(range(1, len(labels) + 1), labels)
#     plt.xlabel('hour')
#     plt.ylabel('incident time')
#     plt.title('Incident time across hours in restday')
#     plt.savefig('./here_incidents_data/incident_time_restday_box_plot.png')
# =============================================================================

# =============================================================================
#     # visualization by using histogram
#     for work_keys in sort_dict_work_lasting_time:
#         plt.figure(dpi=300, figsize=(10,7))
#         plt.hist(sort_dict_work_lasting_time[work_keys], bins=40, facecolor='blue', edgecolor='black')
#         plt.xlabel('incident time')
#         plt.ylabel('frequency')
#         plt.title('Incident lasting time frequency in hour_%s on workday' % (work_keys))
#         plt.savefig('./here_incidents_data/incident_time_hour_%s_workday.png' % (work_keys))
#
#     for rest_keys in sort_dict_rest_lasting_time:
#         plt.figure(dpi=300, figsize=(10,7))
#         plt.hist(sort_dict_rest_lasting_time[rest_keys], bins=40, facecolor='blue', edgecolor='black')
#         plt.xlabel('incident time')
#         plt.ylabel('frequency')
#         plt.title('Incident lasting time frequency in hour_%s on restday' % (rest_keys))
#         plt.savefig('./here_incidents_data/incident_time_hour_%s_restday.png' % (rest_keys))
# =============================================================================
