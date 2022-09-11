from prep_network import Scale
from urbanscales.io.road_network import RoadNetwork
from urbanscales.io.speed_data import SpeedData, Segment
from tqdm import tqdm
from smartprint import smartprint as sprint
import shapely.geometry


class ScaleJF:
    def __init__(self, scale: Scale, speed_data: SpeedData):
        assert scale.RoadNetwork.city_name == speed_data.city_name
        self.bbox_segment_map = {}
        self.Scale = scale
        self.SpeedData = speed_data

        self.set_bbox_segment_map()

    def set_bbox_segment_map(self):
        # Step 1: iterate over segments
        # Step 2: iterate over bboxes
        # Then within the loop populate the dict if both are intersecting

        pbar = tqdm(total=len(self.SpeedData.NID_road_segment_map) * len(self.Scale.dict_bbox_to_subgraph))
        for seg_id in self.SpeedData.NID_road_segment_map:
            seg_poly = (self.SpeedData.NID_road_segment_map[seg_id]).get_shapely_poly()
            for bbox in self.Scale.dict_bbox_to_subgraph.keys():
                N, S, E, W = bbox
                bbox_shapely = shapely.geometry.box(W, S, E, N, ccw=True)
                if seg_poly.intersection(bbox_shapely):
                    if bbox in self.bbox_segment_map:
                        self.bbox_segment_map[bbox].append(Segment.seg_hash(seg_poly))
                    else:
                        self.bbox_segment_map[bbox] = [Segment.seg_hash(seg_poly)]

                pbar.update(1)
        sprint(len(self.bbox_segment_map))


if __name__ == "__main__":
    sd = SpeedData.get_object("Singapore")
    scl = Scale.get_object_at_scale("Singapore", 9)
    scl_jf = ScaleJF(scl, sd)
    print("Nishant")
    debug_stop = 2
