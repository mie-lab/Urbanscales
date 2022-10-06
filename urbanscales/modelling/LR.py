import config
from urbanscales.preprocessing.train_data import TrainDataVectors
import numpy as np
from sklearn.linear_model import LinearRegression
import time
from smartprint import smartprint as sprint


class LR:
    def __init__(self, cityname, scale, tod):
        obj = TrainDataVectors(cityname, scale, tod)

        self.empty_train_data = True

        if not obj.empty_train_data:
            self.empty_train_data = False
            self.X, self.Y = obj.X, obj.Y
            self.compute_score()
        else:
            print("Missing train data")
            self.empty_train_data = True
            pass

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
                        lr_object = LR(city, seed ** depth, tod)
                        if not lr_object.empty_train_data:
                            sprint(lr_object.score)
                        sprint(time.time() - startime)


if __name__ == "__main__":
    LR.compute_scores_for_all_cities()
