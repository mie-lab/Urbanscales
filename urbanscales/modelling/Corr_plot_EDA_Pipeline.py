import csv
import os
import pickle
import shutil
import sys

import matplotlib

# matplotlib.use('TKAgg')

from multiprocessing import Pool
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

# from smartprint import smartprint as sprint
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
        self.scores_MSE = {}
        self.scores_QWK = {}
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
                title=slugify(corr_type + "-corr-" + self.cityname + "-" + str(self.scale) + "-" + str(self.tod)),
                width=1200,
                height=1200,
            )
            # fig.show()

            fig.write_image(
                os.path.join(
                    config.BASE_FOLDER,
                    config.results_folder,
                    slugify(corr_type + "-corr-" + self.cityname + "-" + str(self.scale) + "-" + str(self.tod)),
                )
                + ".png",
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

    def plot_FI_for_trained_model(self, model, X, Y, marker, plot_counter, model_string):
        assert marker in ["train", "val"]
        fname = os.path.join(
            config.BASE_FOLDER,
            config.results_folder,
            slugify(
                "feat-imp-"
                + self.cityname
                + "-"
                + str(self.scale)
                + "-"
                + str(self.tod)
                + "-"
                + model_string
                + marker
                + "-counter"
                + str(plot_counter)
            ),
        )
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

        with open(fname + ".pkl", "wb") as f:
            pickle.dump(fi_dict, f, protocol=config.pickle_protocol)
            print("Pickle of feature importance saved! ")

        list_of_tuples = sorted(fi_dict.items(), key=lambda kv: kv[1][0], reverse=True)
        plt.clf()
        print ("List of tuples: ", list_of_tuples)

        descending_list_of_feature_importance = [ k[0] for k in list_of_tuples ]
        with open(os.path.join(config.BASE_FOLDER, config.results_folder, "feature_importance.csv"), "a") as f2:
            csvwriter = csv.writer(f2)
            csvwriter.writerow([self.cityname, self.scale, self.tod, marker, plot_counter] + descending_list_of_feature_importance)

        column_names_sorted = [x[0] for x in list_of_tuples]
        importance_heights_sorted = [x[1][0] for x in list_of_tuples]
        colorlist_sorted = [x[1][1] for x in list_of_tuples]

        plt.bar(column_names_sorted, importance_heights_sorted, width=0.5, color=colorlist_sorted)
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        plt.savefig(fname + ".png", dpi=300)

        return

    def compute_score(self):
        print(self.X.shape, self.Y.shape)
        if True: # config.ppl_plot_corr; using True for config.ppl_plot_corr now:
            df_temp = pd.DataFrame(self.scale_x(self.X), columns=self.X.columns)
            df_temp["Y"] = self.Y.flatten().tolist()
            self.plot_CORR(df_temp)
            plt.savefig(os.path.join(
                    config.BASE_FOLDER,
                    config.results_folder,
                    "_Corplot-" + self.cityname + "-" + self.scale + str(int(np.random.rand() * 100000000)) +".png",
                ))

    @staticmethod
    def parfunc(params):
        city, seed, depth, tod, model = params
        try:
            lr_object = Pipeline(city, seed ** depth, tod)
        except Exception:
            print ("Error in (city, seed ** depth, tod)", (city, seed ** depth, tod), "\n Skipping ...........")
            return
        if not lr_object.empty_train_data:
            print(model, np.mean(lr_object.scores_MSE[model]), np.mean(lr_object.scores_QWK[model]))
            with open(
                os.path.join(
                    config.BASE_FOLDER,
                    config.results_folder,
                    "_Scores-" + str(int(np.random.rand() * 100000000)) + ".csv",
                ),
                "a",
            ) as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(
                    [
                        slugify(model),
                        city,
                        seed,
                        depth,
                        tod,
                        lr_object.X.shape,
                        np.mean(lr_object.num_train_data_points),
                        np.mean(lr_object.num_test_data_points),
                        np.mean(lr_object.scores_MSE[model]),
                        np.std(lr_object.scores_MSE[model]),
                        np.mean(lr_object.scores_QWK[model]),
                        np.std(lr_object.scores_QWK[model]),
                        len(lr_object.scores_QWK[model]),
                    ]
                )

    @staticmethod
    def compute_scores_for_all_cities():
        with open(os.path.join(config.BASE_FOLDER, config.results_folder, "_Scores_header.csv"), "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(
                [
                    "model",
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
        list_of_models = config.ppl_list_of_NL_models + config.ppl_list_of_baselines

        list_of_params_parallel = []
        for city in config.scl_master_list_of_cities:
            for seed in config.scl_list_of_seeds:
                for depth in config.scl_list_of_depths:
                    for tod in config.td_tod_list:
                        for model in list_of_models:
                            list_of_params_parallel.append((city, seed, depth, tod, model))

        if config.ppl_parallel_overall > 1:
            pool = Pool(config.ppl_parallel_overall)
            print(pool.map(Pipeline.parfunc, list_of_params_parallel))
        else:
            for params in list_of_params_parallel:
                Pipeline.parfunc(params)


if __name__ == "__main__":




    # if config.delete_results_folder and os.path.exists(os.path.join(config.BASE_FOLDER, config.results_folder)):
    #     shutil.rmtree(os.path.join(config.BASE_FOLDER, config.results_folder))
    if not os.path.exists(os.path.join(config.BASE_FOLDER, config.results_folder)):
        os.mkdir(os.path.join(config.BASE_FOLDER, config.results_folder))

    with open(os.path.join(config.BASE_FOLDER, config.results_folder, "feature_importance.csv"), "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["cityname", "scale", "tod", "marker", "plot_counter"] + ["feature" + str(i) for i in range(1, 17)])

    Pipeline.compute_scores_for_all_cities()