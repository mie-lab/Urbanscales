import csv
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.pipeline import make_pipeline

import config
from urbanscales.metrics.QWK import QWK, custom_scoring_QWK
from urbanscales.preprocessing.tile import Tile
from urbanscales.preprocessing.train_data import TrainDataVectors
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import time
from smartprint import smartprint as sprint
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.datasets import load_occupancy
from yellowbrick.model_selection import FeatureImportances
from slugify import slugify
import plotly.express as px
from sklearn.preprocessing import FunctionTransformer


class Pipeline:
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

            if len(config.td_drop_feature_lists) > 0:
                for feat in config.td_drop_feature_lists:
                    self.X.drop(feat, axis=1, inplace=True)

            self.compute_score()
        else:
            print("Missing train data")
            self.empty_train_data = True
            pass

    def plot_CORR(self, df, i):
        for corr_type in ["pearson", "kendall", "spearman"]:
            fig = px.imshow(df.corr(method=corr_type))
            fig.write_image(
                os.path.join(
                    config.results_folder,
                    slugify(
                        corr_type
                        + "-corr-"
                        + config.model
                        + self.cityname
                        + "-"
                        + str(self.scale)
                        + "-"
                        + str(self.tod)
                        + "-counter"
                        + str(i + 1)
                    ),
                )
                + ".png",
            )

    def plot_FI(self, reg, i, x, y):
        plt.clf()
        viz = FeatureImportances(reg, labels=Tile.get_feature_names(), color=["blue"] * self.X.shape[1])
        # , colormap="viridis"

        viz.fit(x, y)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                config.results_folder,
                slugify(
                    "feat-imp-"
                    + config.model
                    + self.cityname
                    + "-"
                    + str(self.scale)
                    + "-"
                    + str(self.tod)
                    + "-counter"
                    + str(i + 1)
                ),
            )
            + ".png",
            dpi=300,
        )

    def compute_score(self):

        sprint(self.X.shape, self.Y.shape)

        for i in range(self.X.shape[0] // config.ppl_smallest_sample * 2):
            x = []
            y = []
            for j in range(self.X.shape[0]):
                if np.random.rand() < config.ppl_smallest_sample / self.X.shape[0]:
                    x.append(self.X.to_numpy()[j, :])
                    y.append(self.Y[j])
            x = np.array(x)
            y = np.array(y)
            sprint(x.shape, y.shape)

            reg = make_pipeline()
            if config.td_standard_scaler:
                reg.steps.append(["standard_scaler", preprocessing.StandardScaler()])
            if config.td_min_max_scaler:
                reg.steps.append(["minmax_scaler", preprocessing.MinMaxScaler()])

            if config.model == "RFR":
                reg.steps.append(["rfr", RandomForestRegressor()])
            elif config.model == "LR":
                reg.steps.append(["lr", LinearRegression()])
            elif config.model == "GBM":
                reg.steps.append(["gbm", GradientBoostingRegressor()])

            self.cv_scores_default += (cross_val_score(reg, x, y, cv=config.ppl_CV_splits)).tolist()
            self.cv_scores_QWK += (
                cross_val_score(reg, x, y, cv=config.ppl_CV_splits, scoring=custom_scoring_QWK)
            ).tolist()

            if config.ppl_plot_FI:
                self.plot_FI(reg[-1], i, pd.DataFrame(x, columns=self.X.columns), pd.DataFrame(y, columns=["Y"]))
            if config.ppl_plot_corr:
                df_temp = pd.DataFrame(x, columns=self.X.columns)
                df_temp["Y"] = y.flatten().tolist()
                self.plot_CORR(df_temp, i)

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
                        lr_object = Pipeline(city, seed ** depth, tod)
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
    shutil.rmtree(config.results_folder)
    os.mkdir(config.results_folder)

    Pipeline.compute_scores_for_all_cities()
