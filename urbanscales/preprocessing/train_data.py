from urbanscales.io.speed_data import SpeedData
from urbanscales.preprocessing.prep_network import Scale
from urbanscales.preprocessing.prep_speed import ScaleJF
from urbanscales.preprocessing.tile import Tile


from urbanscales.io.speed_data import Segment  # this line if not present gives

# an error while depickling a file.


class TrainDataVectors:
    def __init__(self, city_name, scale, tod):
        """

        Args:
            city_name & scale: Trivial
            tod: single number based on granularity
        """
        self.X = []
        self.Y = []
        self.tod = tod
        self.city_name = city_name
        self.scale = scale

        self.set_X_and_Y()

    def set_X_and_Y(self):
        sd = SpeedData.get_object(self.city_name)
        scl = Scale.get_object_at_scale(self.city_name, self.scale)
        # scl_jf = ScaleJF(scl, sd )
        scl_jf = ScaleJF.get_object(self.city_name, self.scale, self.tod)
        assert isinstance(scl_jf, ScaleJF)
        for bbox in scl_jf.bbox_segment_map:
            # assert bbox in scl_jf.bbox_jf_map
            assert isinstance(scl, Scale)
            subg = scl.dict_bbox_to_subgraph[bbox]
            if isinstance(subg, str):
                if subg == "Empty":
                    # we skip creating X and Y for this empty tile
                    # which does not have any roads OR
                    # is outside the scope of the administrative area
                    continue

            assert isinstance(subg, Tile)

            self.X.append(subg.get_vector_of_features())
            self.Y.append(scl_jf.bbox_jf_map[bbox])
        debug_stop = 2


if __name__ == "__main__":
    td = TrainDataVectors("Singapore", 3 ** 2, 3600 * 7 // 600)
    # sg = SpeedData.get_object("Singapore")
    debug_stop = 3
