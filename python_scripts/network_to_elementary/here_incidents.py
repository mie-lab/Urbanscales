import json
import csv


def get_incident_from_json(json_file, incident_file):
    """

    Args:
        json_file: incident file from Here
        incident_file: grade, type, start_time, end_time, o_lat, o_lon, d_lat, d_lon

    Returns:

    """
    # csv file
    with open(incident_file, "w", newline='') as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(['grade', 'type', 'start_time', 'end_time', 'o_lat', 'o_lon', 'd_lat', 'd_lon'])

        # json file
        with open(json_file) as f_json:
            d1 = json.load(f_json)
            d2 = d1['TRAFFICITEMS']['TRAFFICITEM']
            for i in range(len(d2)):
                item = d2[i]

                item_list = [item['CRITICALITY']['DESCRIPTION']]
                item_list.append(item['TRAFFICITEMTYPEDESC'])
                item_list.append(item['STARTTIME'])
                item_list.append(item['ENDTIME'])

                o_loc = item['LOCATION']['GEOLOC']['ORIGIN']
                item_list.append(o_loc['LATITUDE'])
                item_list.append(o_loc['LONGITUDE'])

                d_loc = item['LOCATION']['GEOLOC']['TO']
                for j in range(len(d_loc)):
                    _d_loc = d_loc[j]
                    item_list.append(_d_loc['LATITUDE'])
                    item_list.append(_d_loc['LONGITUDE'])

                csv_writer.writerow(item_list)


if __name__ == '__main__':
    get_incident_from_json(r"..\incident-2022-02-21-18_58_47_minor.json", r"..\test.csv")


