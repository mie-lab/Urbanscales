import sys
import networkx as nx
from osmnx import stats as osxstats
import osmnx as ox

import config


def get_stats_for_one_tile(input):
    """
    overloaded function
    :param if called from this tiles_to_elementary,
        input: tuple of (osm, its corresponding bounding box)

            if called from urban_merge_process,
            input: just the osm tile as a list (basically the sub-graph of polygon)
    :return:
    """

    assert isinstance(input, list)

    if type(input) == 1:
        osm = input[0]
    elif len(input) == 2:
        osm, bbox = input
    else:
        print("Wrong input in length of variables\n in function get_stats_for_one_tile")
        sys.exit(0)

    if osm == "EMPTY":
        stats = "EMPTY_STATS"

    else:
        if config.stats_type == "basic_stats":
            spn = osxstats.count_streets_per_node(osm)
            nx.set_node_attributes(osm, values=spn, name="street_count")
            try:
                stats = ox.basic_stats(osm)
            except:
                print("stats = ox.basic_stats(osm): ", " ERROR\n Probably no edge in graph")
                stats = "EMPTY_STATS"
        else:
            print("WRONG input in stats_type; \n Check config file; exiting execution")
            sys.exit(0)

    if len(input) == 2:
        retval = {bbox: stats}
    elif len(input) == 1:
        retval = stats

    return retval