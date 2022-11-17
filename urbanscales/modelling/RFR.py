import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score, RepeatedKFold
import config
from urbanscales.metrics.QWK import QWK, custom_scoring_QWK
from urbanscales.preprocessing.tile import Tile
from urbanscales.preprocessing.train_data import TrainDataVectors
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
from smartprint import smartprint as sprint
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.datasets import load_occupancy
from yellowbrick.model_selection import FeatureImportances
from slugify import slugify


class RFR:
    def __init__(self, cityname, scale, tod):
        self.cityname, self.scale, self.tod = cityname, scale, tod
        self.cv_scores_default = []
        self.cv_scores_QWK = []

        obj = TrainDataVectors(cityname, scale, tod)

        self.empty_train_data = True

        if not obj.empty_train_data:
            self.empty_train_data = False
            self.X, self.Y = obj.X, obj.Y

            # convert to DF
            self.X = pd.DataFrame(self.X, columns=Tile.get_feature_names())
            # self.Y = pd.DataFrame(self.Y, columns=["JF"])

            self.compute_score()
        else:
            print("Missing train data")
            self.empty_train_data = True
            pass

    def compute_score(self):

        sprint(self.X.shape, self.Y.shape)

        if config.rf_plot_FI:
            plt.clf()
            reg = RandomForestRegressor().fit(self.X, self.Y)
            viz = FeatureImportances(
                reg, labels=Tile.get_feature_names(), color=["blue"] * self.X.shape[1]
            )  # , colormap="viridis"

            viz.fit(self.X, self.Y)
            # viz.show(block=False)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    config.results_folder,
                    slugify("feat-imp-RFR-" + self.cityname + "-" + str(self.scale) + "-" + str(self.tod)),
                )
                + ".png",
                dpi=300,
            )

        for i in range(self.X.shape[0] // config.rf_smallest_sample * 2):
            x = []
            y = []
            for j in range(self.X.shape[0]):
                if np.random.rand() < config.rf_smallest_sample / self.X.shape[0]:
                    x.append(self.X.to_numpy()[j, :])
                    y.append(self.Y[j])
            x = np.array(x)
            y = np.array(y)
            sprint(x.shape, y.shape)
            reg = RandomForestRegressor().fit(self.X, self.Y)
            self.cv_scores_default += (cross_val_score(reg, x, y, cv=config.rf_CV_splits)).tolist()
            self.cv_scores_QWK += (cross_val_score(reg, x, y, cv=config.rf_CV_splits, scoring=custom_scoring_QWK)).tolist()

    @staticmethod
    def compute_scores_for_all_cities():
        with open(os.path.join(config.results_folder, "_RFR_Scores.csv"), "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(
                [
                    "city",
                    "seed",
                    "depth",
                    "tod",
                    "#datapoints",
                    "np.mean(lr_object.cv_scores)",
                    "np.mean(lr_object.cv_scores_QWK)",
                ]
            )

        for city in config.scl_master_list_of_cities:
            for seed in config.scl_list_of_seeds:
                for depth in config.scl_list_of_depths:
                    for tod in config.td_tod_list:
                        sprint(city, seed, depth, tod)
                        startime = time.time()
                        lr_object = RFR(city, seed ** depth, tod)
                        if not lr_object.empty_train_data:
                            sprint(np.mean(lr_object.cv_scores_default), np.mean(lr_object.cv_scores_QWK))

                            with open(os.path.join(config.results_folder, "_RFR_Scores.csv"), "a") as f:
                                csvwriter = csv.writer(f)
                                csvwriter.writerow(
                                    [
                                        city,
                                        seed,
                                        depth,
                                        tod,
                                        lr_object.X.shape,
                                        np.mean(lr_object.cv_scores_default),
                                        np.mean(lr_object.cv_scores_QWK),
                                    ]
                                )
                        # sprint(time.time() - startime)


if __name__ == "__main__":
    RFR.compute_scores_for_all_cities()
