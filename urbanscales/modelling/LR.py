import config
from urbanscales.preprocessing.train_data import TrainDataVectors
import numpy as np
from sklearn.linear_model import LinearRegression
import time
from smartprint import smartprint as sprint


class LR:
    def __init__(self, cityname, scale, tod):
        obj = TrainDataVectors(cityname, scale, tod)
        self.X, self.Y = obj.X, obj.Y
        self.compute_score()

    def compute_score(self):
        reg = LinearRegression().fit(self.X, self.Y)
        self.score = reg.score(self.X, self.Y)

    @staticmethod
    def compute_scores_for_all_cities():
        for city in config.scl_master_list_of_cities:
            for seed in config.scl_list_of_seeds:
                for depth in config.scl_list_of_depths:
                    for tod in config.td_tod_list:
                        sprint(city, seed, depth, tod)
                        startime = time.time()
                        sprint(LR(city, seed ** depth, tod).score)
                        sprint(time.time() - startime)


if __name__ == "__main__":
    LR.compute_scores_for_all_cities()
