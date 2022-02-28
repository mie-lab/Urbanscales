import matplotlib.patches
import osmnx as ox
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import sys
from shapely import geometry
import numpy as np
import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Point, LineString


def get_sg_poly():
    """

    :return:
    """
    geo = {
        "type": "Polygon",
        "coordinates": [
            [
                [103.96078535200013, 1.39109935100015],
                [103.98568769600007, 1.38544342700007],
                [103.99952233200003, 1.38031647300005],
                [104.00342858200003, 1.374172268000066],
                [103.99187259200011, 1.354925848000036],
                [103.97486412900014, 1.334458726000065],
                [103.95435631600009, 1.318101304000052],
                [103.93189537900008, 1.311468817000076],
                [103.90723717500009, 1.308742580000114],
                [103.88770592500003, 1.301255601000136],
                [103.85271243600005, 1.277289130000085],
                [103.84693444100009, 1.271918036000045],
                [103.84408613400012, 1.268500067000034],
                [103.83887780000003, 1.266262111000046],
                [103.82601972700007, 1.264308986000089],
                [103.80160566500007, 1.264797268000081],
                [103.78956139400003, 1.26788971600007],
                [103.78443444100003, 1.273871161000088],
                [103.77588951900009, 1.287583726000108],
                [103.75513756600003, 1.297105210000012],
                [103.73015384200011, 1.302923895000063],
                [103.70875084700003, 1.305243231000119],
                [103.66529381600009, 1.304103908000087],
                [103.6476343110001, 1.308417059000092],
                [103.64039147200003, 1.322251695000091],
                [103.64470462300005, 1.338039455000043],
                [103.67457116000003, 1.38031647300005],
                [103.67888431100005, 1.399237372000073],
                [103.68384850400008, 1.40989817900001],
                [103.69507897200009, 1.421332098000065],
                [103.70834394600013, 1.429388739000089],
                [103.7179468110001, 1.430975653000118],
                [103.73975670700008, 1.428127346000082],
                [103.76221764400009, 1.430975653000118],
                [103.79004967500003, 1.444281317000048],
                [103.80494225400008, 1.448635158000045],
                [103.83155358200003, 1.447088934000092],
                [103.85718834700009, 1.438706773000135],
                [103.93246504000007, 1.401109117000132],
                [103.96078535200013, 1.39109935100015],
            ]
        ],
    }
    poly = Polygon([tuple(l) for l in geo["coordinates"][0]])
    return poly


def get_poly_from_bbox(bbox):
    """

    :param bbox_list_of_lon_lat:
    :return:
    """

    # Our bbox consists of the following format
    # lat1, lon1, lat2, lon2

    coordinates = []
    lat1, lon1, lat2, lon2 = bbox

    # we go counterclockwise
    coordinates.append([lon1, lat1])
    coordinates.append([lon2, lat1])
    coordinates.append([lon2, lat2])
    coordinates.append([lon1, lat2])
    coordinates = [coordinates]

    geo = {
        "type": "Polygon",
        "coordinates": coordinates,
    }
    poly = Polygon([tuple(l) for l in geo["coordinates"][0]])

    return poly


def test_poly_from_bbox():
    bbox = (1.264308986000089, 103.64039147200003, 1.271682032880087, 103.64771697377918)
    poly = get_poly_from_bbox(bbox)
    print(list(poly.exterior.coords))
    assert len(list(poly.exterior.coords)) == 5
    print("Length: ", len(list(poly.exterior.coords)))


if __name__ == "__main__":
    test_poly_from_bbox()
