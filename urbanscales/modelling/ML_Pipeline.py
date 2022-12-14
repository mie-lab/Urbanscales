import csv
import os
import shutil
import sys

import matplotlib

# matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import cm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV, BaseCrossValidator, train_test_split
from sklearn.pipeline import make_pipeline

import config
from urbanscales.metrics.QWK import QWK, custom_scoring_QWK
from urbanscales.preprocessing.tile import Tile
from urbanscales.preprocessing.train_data import TrainDataVectors
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
import time
from smartprint import smartprint as sprint
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.datasets import load_occupancy
from yellowbrick.model_selection import FeatureImportances
from slugify import slugify
import plotly.express as px
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.inspection import permutation_importance


class Pipeline:
    def __init__(self, cityname, scale, tod):
        self.cityname, self.scale, self.tod = cityname, scale, tod
        self.scores_MSE = []
        self.scores_QWK = []
        self.num_test_data_points = []
        self.num_train_data_points = []

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

    def plot_CORR(self, df):
        for corr_type in config.ppl_list_of_correlations:
            fig = px.imshow(
                df.corr(method=corr_type),
                title=slugify(
                    corr_type + "-corr-" + config.model + self.cityname + "-" + str(self.scale) + "-" + str(self.tod)
                ),
                width=1200,
                height=1200,
            )
            # fig.show()

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
                    ),
                )
                + ".png",
            )

    def feature_importance_via_ridge_coefficients(self, x, y, i):
        ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 2, 5, 10]).fit(x, y)
        importance = np.abs(ridge.coef_)

        color = cm.rainbow(np.linspace(0, 1, (self.X.shape[1])))
        colorlist = []
        for j in range(self.X.shape[1]):
            colorlist.append(color[j])

        plt.clf()
        plt.bar(height=importance, x=self.X.columns, color=colorlist)
        plt.ylim(0, 1)

        plt.title("Feature importances via coefficients")
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()
        # plt.show()
        plt.savefig(
            os.path.join(
                config.results_folder,
                slugify(
                    "-FI-via-ridge_coeff-"
                    + self.cityname
                    + "-"
                    + str(self.scale)
                    + "-"
                    + str(self.tod)
                    + "-counter"
                    + str(i + 1)
                ),
            ),
            dpi=300,
        )

    def feature_importance_via_NL_models(self, x, y, i):
        for model_ in config.ppl_list_of_NL_models:
            if model_ == "RFR":
                model = RandomForestRegressor()
            elif model_ == "GBM":
                model = GradientBoostingRegressor()
            param_grid = {
                "n_estimators": [30, 40, 50, 100, 200],
            }

            model = GridSearchCV(model, param_grid=param_grid, cv=config.ppl_CV_splits)
            model.fit(x, y)

            importance = model.best_estimator_.feature_importances_

            color = cm.rainbow(np.linspace(0, 1, (self.X.shape[1])))
            colorlist = []
            for j in range(self.X.shape[1]):
                colorlist.append(color[j])

            # plt.clf()
            plt.bar(height=importance, x=self.X.columns, color=colorlist)
            plt.ylim(0, 1)

            plt.title("Feature importances via importance " + model_)
            plt.xticks(rotation=90, fontsize=7)
            plt.tight_layout()
            # plt.show()
            plt.savefig(
                os.path.join(
                    config.results_folder,
                    slugify(
                        "-FI-via_-"
                        + model_
                        + self.cityname
                        + "-"
                        + str(self.scale)
                        + "-"
                        + str(self.tod)
                        + "-counter"
                        + str(i + 1)
                    ),
                ),
                dpi=300,
            )

    def scale_x(self, x):
        x_trans = np.array(x)
        # 0: None; 1: Divide by Max; 2: StandardScaler(); 3: Divide by max; followed by StandardScaler()
        if config.ppl_scaling_for_EDA == 1:
            max_ = np.max(x_trans)
            x_trans = x_trans / max_
        if config.ppl_scaling_for_EDA == 2:
            x_trans = StandardScaler().fit_transform(x_trans)
        if config.ppl_scaling_for_EDA == 3:
            max_ = np.max(x_trans)
            x_trans = x_trans / max_
            x_trans = StandardScaler().fit_transform(x_trans)
        return x_trans

    def plot_hist(self, df, i):
        df.hist(bins=config.ppl_hist_bins)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                config.results_folder,
                slugify(
                    "-hist-" + self.cityname + "-" + str(self.scale) + "-" + str(self.tod) + "-counter" + str(i + 1)
                ),
            ),
            dpi=300,
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

    def plot_FI_for_trained_model(self, model, X, Y, marker, plot_counter):
        assert marker in ["train", "val"]
        r = permutation_importance(model, X, Y, n_repeats=30)
        colorlist = [
            "#e6194B",
            "#3cb44b",
            "#ffe119",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#42d4f4",
            "#f032e6",
            "#bfef45",
            "#fabed4",
            "#469990",
            "#dcbeff",
            "#9A6324",
            "#fffac8",
            "#800000",
            "#aaffc3",
            "#808000",
            "#ffd8b1",
            "#000075",
            "#a9a9a9",
            "#ffffff",
            "#776600",
        ]
        fi_dict = dict(zip(self.X.columns, zip(r.importances_mean.tolist(), colorlist)))
        list_of_tuples = sorted(fi_dict.items(), key=lambda kv: kv[1][0], reverse=True)
        plt.clf()

        column_names_sorted = [x[0] for x in list_of_tuples]
        importance_heights_sorted = [x[1][0] for x in list_of_tuples]
        colorlist_sorted = [x[1][1] for x in list_of_tuples]

        plt.bar(column_names_sorted, importance_heights_sorted, width=0.5, color=colorlist_sorted)
        plt.xticks(rotation=90, fontsize=8)
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
                    + "-"
                    + marker
                    + "-counter"
                    + str(plot_counter)
                ),
            )
            + ".png",
            dpi=300,
        )

        return

    def compute_score(self):
        sprint(self.X.shape, self.Y.shape)
        if config.ppl_plot_corr:
            df_temp = pd.DataFrame(self.scale_x(self.X), columns=self.X.columns)
            df_temp["Y"] = self.Y.flatten().tolist()
            self.plot_CORR(df_temp)

        range_ = max(self.X.shape[0] // config.ppl_smallest_sample, 1) * 2
        if config.ppl_use_all:
            # Run with full data 7 times
            range_ = 7

        for i in range(range_):
            x = []
            y = []
            for j in range(self.X.shape[0]):
                # sampling without replacement to avoid target leakage
                if (
                    (np.random.rand() < config.ppl_smallest_sample * (1.33) / self.X.shape[0])
                    and not config.ppl_use_all
                ) or config.ppl_use_all:
                    x.append(self.X.to_numpy()[j, :])
                    y.append(self.Y[j])
            x = np.array(x)
            y = np.array(y)

            sprint(x.shape, y.shape)
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
            sprint(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

            self.num_train_data_points.append(X_train.shape[0])
            self.num_test_data_points.append(X_test.shape[0])

            if config.td_min_max_scaler:
                X_train = preprocessing.MinMaxScaler().fit_transform(X_train)
                X_test = preprocessing.MinMaxScaler().fit_transform(X_test)
            if config.td_standard_scaler:
                X_train = preprocessing.StandardScaler().fit_transform(X_train)
                X_test = preprocessing.StandardScaler().fit_transform(X_test)

            reg = make_pipeline()

            if config.model == "RFR":
                reg.steps.append(["rfr", RandomForestRegressor()])
            elif config.model == "LR":
                reg.steps.append(["lr", LinearRegression()])
            elif config.model == "GBM":
                reg.steps.append(["gbm", GradientBoostingRegressor()])
            elif config.model == "RIDGE":
                reg.steps.append(["ridge", Ridge()])
            elif config.model == "LASSO":
                reg.steps.append(["lasso", Lasso()])

            # if config.ppl_hist:
            #     df_temp = pd.DataFrame(self.scale_x(X_train), columns=self.X.columns)
            #     df_temp["Y"] = y_train.flatten().tolist()
            #     self.plot_hist(df_temp, i)
            #
            # if config.ppl_feature_importance_via_coefficients:
            #     self.feature_importance_via_ridge_coefficients(self.scale_x(X_train), y_train, i)
            # if config.ppl_feature_importance_via_NL_models:
            #     self.feature_importance_via_NL_models(self.scale_x(X_train), y_train, i)

            trained_model = reg.fit(pd.DataFrame(X_train, columns=self.X.columns), y_train)
            y_test_predicted = trained_model.predict(pd.DataFrame(X_test, columns=self.X.columns))
            y_test_GT = y_test

            if config.ppl_plot_FI:
                self.plot_FI_for_trained_model(
                    trained_model, pd.DataFrame(X_train, columns=self.X.columns), y_train, "train", i
                )
                self.plot_FI_for_trained_model(
                    trained_model, pd.DataFrame(X_test, columns=self.X.columns), y_test, "val", i
                )

            self.scores_QWK.append(QWK(y_test_GT, y_test_predicted).val)
            self.scores_MSE.append(mean_squared_error(y_test_GT, y_test_predicted))

    @staticmethod
    def compute_scores_for_all_cities():
        with open(os.path.join(config.results_folder, config.model + "_Scores.csv"), "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(
                [
                    "city",
                    "seed",
                    "depth",
                    "tod",
                    "#datapoints",
                    "#datapoints-train-sample-mean",
                    "#datapoints-test-sample-mean",
                    "np.mean(lr_object.scores_MSE)",
                    "np.std(lr_object.scores_MSE)",
                    "np.mean(lr_object.scores_QWK)",
                    "np.std(lr_object.scores_QWK)",
                    "num_splits",
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
                            sprint(np.mean(lr_object.scores_MSE), np.mean(lr_object.scores_QWK))

                            with open(os.path.join(config.results_folder, config.model + "_Scores.csv"), "a") as f:
                                csvwriter = csv.writer(f)
                                csvwriter.writerow(
                                    [
                                        city,
                                        seed,
                                        depth,
                                        tod,
                                        lr_object.X.shape,
                                        np.mean(lr_object.num_train_data_points),
                                        np.mean(lr_object.num_test_data_points),
                                        np.mean(lr_object.scores_MSE),
                                        np.std(lr_object.scores_MSE),
                                        np.mean(lr_object.scores_QWK),
                                        np.std(lr_object.scores_QWK),
                                        len(lr_object.scores_QWK),
                                    ]
                                )
                        # sprint(time.time() - startime)


if __name__ == "__main__":
    shutil.rmtree(config.results_folder)
    os.mkdir(config.results_folder)

    Pipeline.compute_scores_for_all_cities()
