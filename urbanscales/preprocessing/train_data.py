from urbanscales.io.speed_data import SpeedData
from urbanscales.preprocessing.prep_network import Scale
from urbanscales.preprocessing.prep_speed import ScaleJF


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

        self.set_X()
        self.set_Y()

    def set_X(self):
        sd = SpeedData.get_object(self.city_name)
        scl = Scale.get_object_at_scale(self.city_name, self.scale)
        # scl_jf = ScaleJF(scl, sd )
        scl_jf = ScaleJF.get_object_at_scale(self.city_name, self.scale)
