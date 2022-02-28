from get_sg_osm import get_sg_poly, get_poly_from_bbox
from python_scripts.network_to_elementary.osm_to_tiles import fetch_road_network_from_osm_database, split_poly_to_bb


def get_OSM_tiles(bbox_list):
    """

    :param bbox_list:
    :return:
    """
    G_OSM_list = []
    for bbox in bbox_list:
        G_OSM_list.append(
            fetch_road_network_from_osm_database(
                polygon=get_poly_from_bbox(bbox),
                network_type="drive",
                custom_filter=None,
            )
        )

    # '["highway"~"motorway|motorway_link|primary"]'


def osm_to_feature(osm):
    print(osm)


if __name__ == "__main__":
    G_OSM_list = get_OSM_tiles(split_poly_to_bb(get_sg_poly(), 25, plotting_enabled=False))

    for osm in G_OSM_list:
        osm_to_feature(osm)
