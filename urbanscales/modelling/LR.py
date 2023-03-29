import csv
import os

from sklearn.model_selection import cross_val_score

import config
from urbanscales.preprocessing.train_data import TrainDataVectors
import numpy as np
from sklearn.linear_model import LinearRegression
import time

# from smartprint import smartprint as sprint


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
        self.cv_scores = cross_val_score(reg, self.X, self.Y, cv=config.ppl_CV_splits)

    @staticmethod
    def compute_scores_for_all_cities():
        with open(os.path.join(config.BASE_FOLDER, config.results_folder, "_LR_Scores.csv"), "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(["city", "seed", "depth", "tod", "np.mean(lr_object.cv_scores)"])

        for city in config.scl_master_list_of_cities:
            for seed in config.scl_list_of_seeds:
                for depth in config.scl_list_of_depths:
                    for tod in config.td_tod_list:
                        print(city, seed, depth, tod)
                        startime = time.time()
                        lr_object = LR(city, seed ** depth, tod)
                        if not lr_object.empty_train_data:
                            print(np.mean(lr_object.cv_scores))
                            with open(
                                os.path.join(config.BASE_FOLDER, config.results_folder, "_LR_Scores.csv"), "a"
                            ) as f:
                                csvwriter = csv.writer(f)
                                csvwriter.writerow([city, seed, depth, tod, np.mean(lr_object.cv_scores)])
                        # print (time.time() - startime)


if __name__ == "__main__":
    LR.compute_scores_for_all_cities()
