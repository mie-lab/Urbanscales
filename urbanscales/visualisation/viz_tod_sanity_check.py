import csv
import os
print (os.getcwd())

ZIPPEDfoldername = "train_data_three_scales_mean"
os.system("tar -xf " + ZIPPEDfoldername + ".tar.gz")


import pickle
import copy
import shutil
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import cm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from smartprint import smartprint as sprint
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
import time

# matplotlib.use('TKAgg')

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import config
from urbanscales.metrics.QWK import QWK
from urbanscales.preprocessing.train_data import TrainDataVectors
import numpy as np
from sklearn.linear_model import RidgeCV

# from smartprint import smartprint as sprint
from slugify import slugify
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


# All custom unpicklers are due to SO user Pankaj Saini's answer:  https://stackoverflow.com/a/51397373/3896008
class CustomUnpicklerTrainDataVectors(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module, name)


class SanityCheckTOD:
    def __init__(self, fname):
        self.scores_MSE = {}
        self.scores_QWK = {}
        self.num_test_data_points = []
        self.num_train_data_points = []

        temp = copy.deepcopy(CustomUnpicklerTrainDataVectors(open(fname, "rb")).load())
        self.__dict__.update(temp.__dict__)
        nparrayX = np.array(self.X)
        nparrayY = np.array(self.Y)
        print(nparrayX.shape, nparrayY.shape)




if __name__ == "__main__":
    for scale in [25, 50, 100]:
        for city in "Singapore|Zurich|Mumbai|Auckland|Istanbul|MexicoCity|" \
                    "Bogota|NewYorkCity|Capetown|London".split("|"):
            a = []
            for tod in range(24):
                # Format: "_scale_25_train_data_4"
                sprint (os.getcwd())
                fname = ZIPPEDfoldername + "/" +city+ "/_scale_" + str(scale) + "_train_data_" + str(tod) + \
                        ".pkl"
                try:
                    obj = SanityCheckTOD(fname)
                except FileNotFoundError:
                    print ("Error in fname: ", fname)
                    continue
                a.append(obj.Y.sum())
            plt.plot(a, label="City: " + city)
        plt.title("Scale: " + str(scale))
        plt.legend(fontsize=8, loc="best")
        plt.savefig("Mean_case_" + str(scale) + ".png", dpi=300)
        plt.show()

    os.system("rm -rf " + ZIPPEDfoldername)




    #
    # def basic_correlation_analysis(self, common_features):
    #     correlations = []
    #     p_values = []
    #     r2_values = []  # List to store R^2 values
    #
    #     feature_names = list(common_features)
    #
    #     if len(self.nparrayY.shape) == 1:
    #         for i, feature_name in enumerate(feature_names):
    #             feature_i = self.nparrayX[:, i]
    #             corr_coefficient, p_value = pearsonr(feature_i, self.nparrayY)
    #             correlations.append(corr_coefficient)
    #             p_values.append(p_value)
    #
    #             # Calculate R^2 value for the feature
    #             model = LinearRegression().fit(feature_i.reshape(-1, 1), self.nparrayY)
    #             r2_values.append(model.score(feature_i.reshape(-1, 1), self.nparrayY))
    #
    #         sorted_indices = np.argsort(correlations)[::-1]
    #         sorted_correlations = [correlations[i] for i in sorted_indices]
    #         sorted_p_values = [p_values[i] for i in sorted_indices]
    #         sorted_r2_values = [r2_values[i] for i in sorted_indices]
    #         sorted_feature_names = [feature_names[i] for i in sorted_indices]
    #
    #         plt.figure(figsize=(15, 7))
    #         bars = plt.bar(sorted_feature_names, sorted_correlations, color='c', label="Pearson Correlation")
    #
    #         for bar, p_value in zip(bars, sorted_p_values):
    #             if p_value > 0.05:
    #                 bar.set_color('r')
    #
    #         # Displaying R^2 values on top of the bars
    #         for i, r2_value in enumerate(sorted_r2_values):
    #             plt.text(i, sorted_correlations[i] + 0.01, f"R^2: {r2_value:.2f}", ha='center', rotation=90,
    #                      color='blue', fontsize=8)
    #
    #         plt.xlabel('Feature Names')
    #         plt.ylabel('Pearson Correlation Coefficient')
    #         plt.title(f'Correlation Coefficients between Features of X and Y')
    #         plt.xticks(rotation=45, ha="right")
    #
    #         ax = plt.gca()
    #         bars = ax.bar(sorted_feature_names, sorted_correlations, color=color, label=f"Scale {color}")
    #         # plt.grid(axis='y')
    #         # plt.legend()
    #         # plt.tight_layout()
    #         # plt.show()
    #
    #     else:
    #         print("Y is not 1D and cannot be correlated directly with X.")
