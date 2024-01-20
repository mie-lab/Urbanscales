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
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import scipy.stats as stats

current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.join(current_dir, '..')
os.chdir(current_dir)
sprint (os.getcwd())

ZIPPEDfoldername = "train_data_three_scales_max"
PLOTS_foldername = "max_case_corr_analysis/"

if "mean" in ZIPPEDfoldername:
    assert "mean" in PLOTS_foldername
    MEAN_MAX = "mean"
if "max" in ZIPPEDfoldername:
    assert "max" in PLOTS_foldername
    MEAN_MAX = "max"

os.system("tar -xf ../" + ZIPPEDfoldername + ".tar.gz")

class CustomUnpicklerTrainDataVectors(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module, name)

class RandomForestAnalysis:
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

    def random_forest_feature_importance_with_CI(self, common_features, scaling=False, significance_level=0.05,
                                                 list_of_scales=[25, 50, 100],
                                                 city="Singapore",
                                                 tod = "1-24",
                                                 ):
        scales = "-".join([str(x) for x in list_of_scales])

        n_splits = 10
        kfold = KFold(n_splits=n_splits, shuffle=False) #, random_state=42)
        feature_importances_distributions = np.zeros((len(common_features), n_splits))

        val_error_accumulator = []
        for i, (train_index, test_index) in enumerate(kfold.split(self.nparrayX, self.nparrayY)):
            X_train, X_test = self.nparrayX[train_index], self.nparrayX[test_index]
            Y_train, Y_test = self.nparrayY[train_index], self.nparrayY[test_index]

            # Scale the data if scaling is True
            if scaling:
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            rf = RandomForestRegressor(n_estimators=200)
            rf.fit(X_train, Y_train)

            # Store the feature importances for each fold
            feature_importances_distributions[:, i] = rf.feature_importances_

            # Accumulate validation errors
            val_error_accumulator.append(np.mean((Y_test - rf.predict(X_test)) ** 2))

        # Calculate mean, standard deviation, and confidence intervals of feature importances
        mean_importances = np.mean(feature_importances_distributions, axis=1)
        std_importances = np.std(feature_importances_distributions, axis=1)
        ci_bounds = stats.norm.interval(0.95, loc=mean_importances, scale=std_importances / np.sqrt(n_splits))

        mean_importances = np.mean(feature_importances_distributions, axis=1)
        std_importances = np.std(feature_importances_distributions, axis=1)
        ci_bounds = stats.norm.interval(0.95, loc=mean_importances, scale=std_importances / np.sqrt(n_splits))

        t_values, p_values = stats.ttest_1samp(feature_importances_distributions, 0, axis=1)
        significant_features = p_values < significance_level

        # plt.figure(figsize=(10, 6))
        for idx, is_significant in enumerate(significant_features):
            color = 'blue' if is_significant else 'lightgrey'
            alpha = 1.0 if is_significant else 0.5
            plt.errorbar(idx, mean_importances[idx], yerr=[[mean_importances[idx] - ci_bounds[0][idx]], [ci_bounds[1][idx] - mean_importances[idx]]],
                         fmt='o', ecolor='r', capsize=5, color=color, alpha=alpha)

        plt.xticks(range(len(common_features)), common_features, rotation=90)
        plt.title(f'Feature Importances for {city} at Scale {scales}' + " tod " + tod)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(f'no_total_yes_metered_FI for {city} at Scale {scales}' + " tod " + tod+'.png', dpi=300)
        plt.show(block=False)


    # return mean_importances, ci_bounds, std_importances, significant_features, np.mean(val_error_accumulator), rf

    def random_forest_feature_importance(self, common_features, scaling=False):
        n_splits = 10

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        total_feature_importances = np.zeros(len(common_features))

        val_error_accumulator = []
        for train_index, test_index in kfold.split(self.nparrayX, self.nparrayY):
            X_train, X_test = self.nparrayX[train_index], self.nparrayX[test_index]
            Y_train, Y_test = self.nparrayY[train_index], self.nparrayY[test_index]

            # Scale the training data and transform the test data using the same scaler
            scaler = MinMaxScaler()
            if scaling:
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            rf = RandomForestRegressor(n_estimators=200)
            rf.fit(X_train, Y_train)
            total_feature_importances += rf.feature_importances_
            val_error_accumulator.append(np.mean((Y_test - rf.predict(X_test)) ** 2))

        total_feature_importances /= n_splits
        return total_feature_importances, np.mean(val_error_accumulator), rf

    def plot_feature_importances(self, importances, scale, common_features, list_of_cities):
        # plt.clf()
        plt.title(f"Feature Importances RF " + list_of_cities)
        plt.plot(range(len(common_features)), importances, marker='o', linestyle='-', label=f"Scale: {scale}")
        plt.xticks(range(len(common_features)), common_features, rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.legend()





    def plot_partial_dependence_rf(self, common_features, list_of_cities, trained_model, X_scaled, Scale, plotting_in_this_func=False):
        """
        returns dict_features_plot_x_y_data["-".join(list_of_cities), feature_name, Scale] = {
                "x": pd_results["grid_values"][0],
                "y": pd_results["average"][0],
            }
        if plotting plotting_in_this_func == False";

        if plotting plotting_in_this_func == True";
                            just plots and saves them
        """
        rf = trained_model
        dict_features_plot_x_y_data = {}

        X_scaled = pd.DataFrame(X_scaled, columns=common_features)

        # Extracting feature indices from common_features
        feature_indices = [X_scaled.columns.tolist().index(feature) for feature in common_features]

        # Plotting PDP
        # Compute the partial dependence values

        dict_features_plot_x_y_data = {}
        for feature_index in tqdm(feature_indices, desc="Creating PDP"):
            plt.clf()
            # Compute the partial dependence values for individual features
            pd_results = partial_dependence(rf, X_scaled, features=[feature_index], grid_resolution=100)
            feature_name = common_features[feature_index]

            if plotting_in_this_func:

                assert common_features[feature_index] == X_scaled.columns[feature_index]

                deciles = {0: np.linspace(np.min(X_scaled[feature_name]), np.max(X_scaled[feature_name]), num=100)}
                display = PartialDependenceDisplay(
                    [pd_results],
                    features=[(feature_index,)],
                    feature_names=common_features,
                    target_idx=0,
                    deciles=deciles
                )
                display.plot()
                plt.title(f"Partial Dependence Plots RF for {list_of_cities}" + "_" + feature_name)
                plt.tight_layout()
                plt.savefig( MEAN_MAX + "_case_PDP/" + "-".join(list_of_cities) + "PDP-" + feature_name + "-scale-" +str(Scale)
                            + ".png", dpi=300)
                plt.clf()

            dict_features_plot_x_y_data["-".join(list_of_cities), feature_name, Scale] = {
                "x": pd_results["grid_values"][0],
                "y": pd_results["average"][0],
            }
        return dict_features_plot_x_y_data


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
    plot_data_DICT = {}
    # tod_list_of_list = [
    #     list(range(24))
    #     ]

    # common_features = None
    # for list_of_cities in list_of_cities_list_of_list:
    #     for tod_list in tod_list_of_list:
    #         for scale in [
    #             25,
    #             50,
    #             100
    #         ]:
    #             for city in list_of_cities:
    #                 for tod in tod_list:
    #                     fname = f"{ZIPPEDfoldername}/{city}/_scale_{scale}_train_data_{tod}.pkl"
    #                     try:
    #                         temp_obj = CustomUnpicklerTrainDataVectors(open(fname, "rb")).load()
    #                         if isinstance(temp_obj.X, pd.DataFrame):
    #                             features_in_current_file = set(temp_obj.X.columns.tolist())
    #                             common_features = common_features.intersection(
    #                                 features_in_current_file) if common_features else features_in_current_file
    #                     except FileNotFoundError:
    #                         print("Error in fname:", fname)
    # common_features = list(set(common_features))
    # list.sort(common_features)

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
        # 'total_crossings'
    ]


    for list_of_cities in list_of_cities_list_of_list:
        for tod_list in tod_list_of_list:
            plt.clf()
            # plt.figure(figsize=(12, 6))
            for scale in [
                25,
                # 50,
                # 100
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


                areas = {
                    'betweenness':1,
                    'circuity_avg':1,
                    'global_betweenness':1,
                    'k_avg':1,
                    'lane_density':1,
                    # 'm',
                    # 'n',
                    'metered_count':1, #(50*50/(scale*scale)),
                    # 'n',
                    'non_metered_count':1, #(50*50/(scale*scale)),
                    'street_length_total':1 ,#(50*50/(scale*scale)),
                    'streets_per_node_count_5':1, #(50*50/(scale*scale)),
                    'total_crossings':1 ,# (50*50/(scale*scale))
                }

                df = pd.DataFrame(combined_X, columns=common_features)
                for feature in common_features:
                    df[feature] = df[feature]/ areas[feature]
                combined_X = df.to_numpy()

                corr_obj = RandomForestAnalysis(combined_X, combined_Y)

                corr_obj.random_forest_feature_importance_with_CI(common_features, scaling=True,
                                                                  significance_level=0.05,
                                                                  list_of_scales=[scale],
                                                                  city="-".join(list_of_cities),
                                                                  tod = "-".join([str(x) for x in tod_list])
                                                                  )
