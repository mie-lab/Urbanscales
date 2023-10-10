import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append("../../../")
from urbanscales.preprocessing.train_data import TrainDataVectors
import pickle
import copy
from sklearn.model_selection import KFold
from smartprint import smartprint as sprint

current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.join(current_dir, '..')
os.chdir(current_dir)
sprint (os.getcwd())

ZIPPEDfoldername = "train_data_three_scales_max"
PLOTS_foldername = "max_case_corr_analysis/"

if "mean" in ZIPPEDfoldername:
    assert "mean" in PLOTS_foldername
if "max" in ZIPPEDfoldername:
    assert "max" in PLOTS_foldername

os.system("tar -xf ../" + ZIPPEDfoldername + ".tar.gz")

class CustomUnpicklerTrainDataVectors(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module, name)

class CorrelationAnalysis:
    def __init__(self, X_data=None, Y_data=None, fname=None):
        if fname:
            temp = copy.deepcopy(CustomUnpicklerTrainDataVectors(open(fname, "rb")).load())
            self.__dict__.update(temp.__dict__)
            self.nparrayX = np.array(self.X)
            self.nparrayY = np.array(self.Y)
            self.columns = self.X.columns.tolist()
        elif X_data is not None and Y_data is not None:
            if isinstance(X_data, pd.DataFrame):
                self.nparrayX = X_data.to_numpy()
                self.columns = X_data.columns.tolist()
            else:  # assume it's a numpy array
                self.nparrayX = X_data
                self.columns = [f"feature_{i}" for i in range(X_data.shape[1])]  # generic names
            self.nparrayY = Y_data
        else:
            raise ValueError("Either provide a filename or both X_data and Y_data arrays for initialization.")
        print(self.nparrayX.shape, self.nparrayY.shape)


    def random_forest_feature_importance(self, common_features):
        n_splits = 7

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        total_feature_importances = np.zeros(len(common_features))

        for train_index, test_index in kfold.split(self.nparrayX, self.nparrayY):
            X_train, X_test = self.nparrayX[train_index], self.nparrayX[test_index]
            Y_train, Y_test = self.nparrayY[train_index], self.nparrayY[test_index]
            rf = RandomForestRegressor(n_estimators=100) # , random_state=42)
            rf.fit(X_train, Y_train)
            total_feature_importances += rf.feature_importances_
        total_feature_importances /= n_splits

        return total_feature_importances

    def plot_feature_importances(self, importances, scale, common_features, list_of_cities):
        plt.title(f"Feature Importances RF " + list_of_cities)
        plt.plot(range(len(common_features)), importances, marker='o', linestyle='-', label=f"Scale: {scale}")
        plt.xticks(range(len(common_features)), common_features, rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.legend()


if __name__ == "__main__":
    list_of_cities = "Singapore|Zurich|Mumbai|Auckland|Istanbul|MexicoCity|Bogota|NewYorkCity|Capetown|London".split("|")
    list_of_cities_list_of_list = [
                                    # list_of_cities[:2],
                                    # list_of_cities[2:]
                                    [list_of_cities[0]],
                                    [list_of_cities[1]],
                                    [list_of_cities[2]],
                                    [list_of_cities[3]],
                                    [list_of_cities[4]],
                                    [list_of_cities[5]],
                                    [list_of_cities[6]],
                                    [list_of_cities[7]],
                                    [list_of_cities[8]],
                                    [list_of_cities[9]],

                                ]

    tod_list_of_list = [
        # [6,7,8,9],
        # [0,1,2,3,4,5],
        # [10,11,12,13,14,15],
        # [16,17,18,19],
        # [20,21,22,23]
     range(24),
        ]

    # tod_list_of_list = [
    #     list(range(24))
    #     ]

    common_features = None
    for list_of_cities in list_of_cities_list_of_list:
        for tod_list in tod_list_of_list:
            for scale in [
                25,
                50,
                100
            ]:
                for city in list_of_cities:
                    for tod in tod_list:
                        fname = f"{ZIPPEDfoldername}/{city}/_scale_{scale}_train_data_{tod}.pkl"
                        try:
                            temp_obj = CustomUnpicklerTrainDataVectors(open(fname, "rb")).load()
                            if isinstance(temp_obj.X, pd.DataFrame):
                                features_in_current_file = set(temp_obj.X.columns.tolist())
                                common_features = common_features.intersection(
                                    features_in_current_file) if common_features else features_in_current_file
                        except FileNotFoundError:
                            print("Error in fname:", fname)
    common_features = list(set(common_features))
    list.sort(common_features)
    common_features = [
        'betweenness',
        'circuity_avg',
        'global_betweenness',
        'k_avg',
        'lane_density',
        # 'm',
        'metered_count',
        # 'n',
        'non_metered_count',
        'street_length_total',
        'streets_per_node_count_5',
        'total_crossings'
    ]

    for list_of_cities in list_of_cities_list_of_list:
        for tod_list in tod_list_of_list:
            plt.figure(figsize=(12, 6))
            for scale in [
                25,
                50,
                100
            ]:


                combined_X_for_scale, combined_Y_for_scale = [], []
                for city in list_of_cities:
                    combined_X_for_city, combined_Y_for_city = [], []
                    for tod in tod_list:

                        fname = f"{ZIPPEDfoldername}/{city}/_scale_{scale}_train_data_{tod}.pkl"
                        try:
                            temp_obj = CustomUnpicklerTrainDataVectors(open(fname, "rb")).load()
                            if isinstance(temp_obj.X, pd.DataFrame):
                                filtered_X = temp_obj.X[list(common_features)]
                                combined_X_for_city.append(filtered_X)
                                combined_Y_for_city.append(temp_obj.Y)
                        except FileNotFoundError:
                            print("Error in fname:", fname)

                    combined_X_for_scale.append(np.vstack(combined_X_for_city))
                    combined_Y_for_scale.append(np.hstack(combined_Y_for_city))

                combined_X = np.vstack(combined_X_for_scale)
                combined_Y = np.hstack(combined_Y_for_scale)
                corr_obj = CorrelationAnalysis(combined_X, combined_Y)
                importances = corr_obj.random_forest_feature_importance(common_features)
                indices = np.argsort(importances)[::-1]
                corr_obj.plot_feature_importances(importances, scale, list(common_features), "-".join(list_of_cities))

            # plt.legend()
            # plt.ylim(0.5)
            plt.tight_layout()
            plt.savefig("-".join(list_of_cities) + "-RF-reduced_features-" + ZIPPEDfoldername + "tod" +
                        "-".join([str(s) for s in tod_list]) + ".png")
            # plt.show()
