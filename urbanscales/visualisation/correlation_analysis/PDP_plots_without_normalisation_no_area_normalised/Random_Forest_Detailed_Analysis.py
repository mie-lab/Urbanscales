import csv
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

        common_features_colors = {
            'betweenness': '#1f77b4',  # blue
            'circuity_avg': '#ff7f0e',  # orange
            'global_betweenness': '#2ca02c',  # green
            'k_avg': '#d62728',  # red
            'lane_density': '#9467bd',  # purple
            'metered_count': '#8c564b',  # brown
            'non_metered_count': '#e377c2',  # pink
            'street_length_total': '#7f7f7f',  # gray
            'm': '#bcbd22',  # lime
            'total_crossings': '#17becf',  # cyan
            "n": "black"
        }
        plt.clf()
        scales = "-".join([str(x) for x in list_of_scales])

        n_splits = 7
        kfold = KFold(n_splits=n_splits, shuffle=False) #, random_state=42)
        feature_importances_distributions = np.zeros((len(common_features), n_splits))

        val_error_accumulator = []
        rf_list = []
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

            rf_list.append(rf)

            # Store the feature importances for each fold
            feature_importances_distributions[:, i] = rf.feature_importances_

            # Accumulate validation errors
            val_error_accumulator.append(np.mean((Y_test - rf.predict(X_test)) ** 2))

        # Initialize dictionary to store partial dependence data
        partial_dep_data = {feature: [] for feature in common_features}
        partial_dep_x_axis = {feature: [] for feature in common_features}

        n_points = 100

        # Calculate partial dependence for each model and feature
        for counter, rf in enumerate(rf_list):
            x_train_df = pd.DataFrame(X_train, columns=common_features)
            for feature in common_features:
                pdp_output = partial_dependence(rf, x_train_df, [feature], grid_resolution=n_points)
                grid, partial_dep = pdp_output['grid_values'][0], pdp_output['average'][0]
                partial_dep_data[feature].append(partial_dep)
                if counter == 0:
                    partial_dep_x_axis[feature].append(grid)

        # Process the results and plot
        # plt.figure(figsize=(12, 3 * len(common_features)))
        plt.clf()
        for i, feature in enumerate(common_features):
            pdps = np.array(partial_dep_data[feature])
            pdp_median = np.median(pdps, axis=0)
            pdp_mean = np.mean(pdps, axis=0)
            pdp_min = np.min(pdps, axis=0)
            pdp_max = np.max(pdps, axis=0)

            grid = partial_dep_x_axis[feature][0]
            # for each_pdp in pdps:
            #     plt.plot(partial_dep_x_axis[], each_pdp, alpha=0.2, color="tab:blue")

            # plt.plot(grid, pdp_median, label="Median PDP " + feature, color=common_features_colors[feature], linewidth=1.5)
            plt.plot(grid, pdp_mean, label=feature, color=common_features_colors[feature], linewidth=1.5)
            with open("PDP-" + f'MEAN_MAX_{MEAN_MAX}_FI_Mean_{city}_Scale_{scales}_TOD_{tod}_FEATURE_{feature}' + "csv", "w") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(["grid"] + grid.tolist())
                csvwriter.writerow(["PDP"] + pdp_mean.tolist())

            # plt.fill_between(grid, pdp_min, pdp_max, alpha=0.2, label="Min-Max Range")
            # plt.title(f'Partial Dependence for {feature}')
        plt.legend(ncol=2)
        # plt.title(f'MEAN_MAX_{MEAN_MAX}_FI_Mean_{city}_Scale_{scales}')
        plt.title("Mean PDP across CV")
        plt.tight_layout()
        plt.ylim(0, 10)
        plt.savefig('pdp_CV_mean_aggregated-' f'MEAN_MAX_{MEAN_MAX}_FI_Mean_{city}_Scale_{scales}_TOD_{tod}.png')
        plt.show(block=False)


        # Process the results and plot
        # plt.figure(figsize=(12, 3 * len(common_features)))
        # for i, feature in enumerate(common_features):
        #     pdps = np.array(partial_dep_data[feature])
        #     pdp_median = np.median(pdps, axis=0)
        #     pdp_mean = np.mean(pdps, axis=0)
        #     pdp_min = np.min(pdps, axis=0)
        #     pdp_max = np.max(pdps, axis=0)
        #
        #     grid = partial_dep_x_axis[feature][0]
        #     # for each_pdp in pdps:
        #     #     plt.plot(partial_dep_x_axis[], each_pdp, alpha=0.2, color="tab:blue")
        #
        #     plt.plot(grid, pdp_median, label= feature, color=common_features_colors[feature], linewidth=1.5)
        #     # plt.plot(grid, pdp_mean, label="Mean PDP " + feature, color=common_features_colors[feature], linewidth=1.5)
        #
        #     # plt.fill_between(grid, pdp_min, pdp_max, alpha=0.2, label="Min-Max Range")
        #     # plt.title(f'Partial Dependence for {feature}')
        # plt.legend(ncol=2)
        # plt.title("Median PDP across CV")
        # plt.tight_layout()
        # plt.ylim(0, 10)
        # plt.savefig('pdp_CV_median_aggregated-' f'MEAN_MAX_{MEAN_MAX}_FI_Mean_{city}_Scale_{scales}_TOD_{tod}.png')
        # plt.show(block=False)
        # sys.exit(0)

        # Calculate mean, standard deviation, and confidence intervals of feature importances

        mean_importances = np.mean(feature_importances_distributions, axis=1)
        std_importances = np.std(feature_importances_distributions, axis=1)
        ci_bounds = stats.norm.interval(0.95, loc=mean_importances, scale=std_importances / np.sqrt(n_splits))

        # Creating a DataFrame to store the mean feature importances
        df = pd.DataFrame({
            'scale': scales,
            'city': city,
            'tod': tod,
            'FI_mean': mean_importances
        }, index=common_features)

        # Saving the DataFrame to a CSV file
        csv_filename = f'MEAN_MAX_{MEAN_MAX}_FI_Mean_{city}_Scale_{scales}_TOD_{tod}.csv'
        df.to_csv(csv_filename)

        t_values, p_values = stats.ttest_1samp(feature_importances_distributions, 0, axis=1)
        significant_features = p_values < significance_level

        # plt.figure(figsize=(10, 6))
        plt.clf()
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
        plt.savefig(f'MEAN_MAX_{MEAN_MAX}_Feature Importances for {city} at Scale {scales}' + " tod " + tod+'.png', dpi=300)
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

def plot_feature_dist(X, scale, city, tod, common_features):
    print (X)

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
        'm',
        'metered_count',
        'n',
        'non_metered_count',
        'street_length_total',
        # 'streets_per_node_count_5',
        'total_crossings',

    ]


    for list_of_cities in list_of_cities_list_of_list:
        for tod_list in tod_list_of_list:
            plt.clf()
            # plt.figure(figsize=(12, 6))
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


                areas = {
                    'betweenness':1,
                    'circuity_avg':1,
                    'global_betweenness':1,
                    'k_avg':1,
                    'lane_density':1,
                    'm':1,
                    'n':1,
                    'metered_count':1, # (10000/(scale*scale)),
                    'non_metered_count':1, # (100*100/(scale*scale)),
                    'street_length_total':1, # (100*100/(scale*scale)),
                    'streets_per_node_count_5':1, # (100*100/(scale*scale)),
                    'total_crossings':1, # 100*100/(scale*scale),

                }

                df = pd.DataFrame(combined_X, columns=common_features)
                for feature in common_features:
                    df[feature] = df[feature]/ areas[feature]
                combined_X = df.to_numpy()


                corr_obj = RandomForestAnalysis(combined_X, combined_Y)
                corr_obj.random_forest_feature_importance_with_CI(common_features, scaling=False,
                                                                  significance_level=0.05,
                                                                  list_of_scales=[scale],
                                                                  city="-".join(list_of_cities),
                                                                  tod = "-".join([str(x) for x in tod_list])
                                                                  )


                # plot_feature_dist(combined_X, scale=str(scale),
                #                         city="-".join(list_of_cities),
                #                         tod = "-".join([str(x) for x in tod_list]),
                #                         common_features=common_features)

"""
                scaling = True
                if scaling:
                    importances, GoF, trained_model = corr_obj.random_forest_feature_importance(common_features,
                                                                                                scaling=True)
                    sprint(scale, GoF, "With scaling")
                else:
                    importances, GoF, trained_model = corr_obj.random_forest_feature_importance(common_features,
                                                                                                scaling=False)
                    sprint (scale, GoF, "No scaling")
                

       
 


                # indices = np.argsort(importances)[::-1]

                # corr_obj.plot_feature_importances(importances, scale, list(common_features), "-".join(list_of_cities))
                    
                if scaling:
                    # scaled_X = StandardScaler().fit_transform(combined_X)
                    # scaled_X = PowerTransformer().fit_transform(combined_X)
                    scaled_X = MinMaxScaler().fit_transform(combined_X)
                
                else:
                    scaled_X = combined_X
                plot_data_DICT.update(corr_obj.plot_partial_dependence_rf(common_features, list_of_cities, trained_model,
                                                    scaled_X, scale))



        # plt.tight_layout()
        # plt.savefig("max_case_FI_area_normalised/" + f"Feature Importances RF " + list_of_cities[0] + "scales" + ".png")



    list_of_scales = [25, 50, 100]  # Add other scales if needed, e.g., [25, 50, 100]
    for list_of_cities in list_of_cities_list_of_list:
        for feature in common_features:
            # Plot data for each scale on the same plot
            for scale in list_of_scales:
                city = "-".join(list_of_cities)
                if (city, feature, scale) in plot_data_DICT:
                    max_y = np.max(plot_data_DICT[city, feature, scale]["y"])
                    max_x = np.max(plot_data_DICT[city, feature, scale]["x"])
                    min_y = np.min(plot_data_DICT[city, feature, scale]["y"])
                    min_x = np.min(plot_data_DICT[city, feature, scale]["x"])
                    plt.plot((plot_data_DICT[city, feature, scale]["x"] - min_x)/(max_x - min_x),
                             (plot_data_DICT[city, feature, scale]["y"] - min_y)/(max_y - min_y),
                             label=f"{feature} Scale@{scale}")

            # After adding all scales to the plot, finalize the plot
            plt.ylim(-0.02, 1.05)
            plt.xlim(-0.02, 1.05)
            plt.legend()
            plt.title(f"{city} - {feature}")
            plt.savefig(MEAN_MAX + "_case_PDP/X-scaled-"+str(scaling)+"-normalised-" + city + "-PDP-" + feature + ".png", dpi=300)
            plt.clf()  # Clear the current figure


    list_of_scales = [25, 50, 100]  # Add other scales if needed, e.g., [25, 50, 100]
    dict_city_scale_importance = {}
    for list_of_cities in list_of_cities_list_of_list:
        for feature in common_features:
            # Plot data for each scale on the same plot
            for scale in list_of_scales:
                city = "-".join(list_of_cities)
                if (city, feature, scale) in plot_data_DICT:
                    plt.plot(plot_data_DICT[city, feature, scale]["x"],
                             plot_data_DICT[city, feature, scale]["y"],
                             label=f"{feature} Scale@{scale} FI:" + str(round(np.std(plot_data_DICT[city, feature, scale]["y"]),2)) )
                    dict_city_scale_importance[city, feature, scale] = np.std(plot_data_DICT[city, feature, scale]["y"])


            # After adding all scales to the plot, finalize the plot
            # plt.ylim(0, 1)
            # plt.xlim(0, 1)
            plt.legend()
            plt.title(f"{city} - {feature}")
            plt.savefig(MEAN_MAX + "_case_PDP/X_scaled-"+str(scaling)+"-unnormalised-" + city + "-PDP-" + feature + ".png", dpi=300)
            plt.clf()  # Clear the current figure


    city_colors = {
        "Singapore": "#FF5733",  # Red
        "Zurich": "#3498DB",  # Blue
        "Mumbai": "#2ECC71",  # Green
        "Auckland": "#F1C40F",  # Yellow
        "Istanbul": "#9B59B6",  # Purple
        "MexicoCity": "#E74C3C",  # Orange
        "Bogota": "#34495E",  # Dark Blue
        "NewYorkCity": "#1ABC9C",  # Teal
        "Capetown": "#7D3C98",  # Plum
        "London": "#D35400"  # Pumpkin
    }

    for feature in common_features:
        plt.clf()
        for list_of_cities in list_of_cities_list_of_list:
            city = "-".join(list_of_cities)
            importances = [dict_city_scale_importance[city, feature, scale] for scale in list_of_scales]
            if importances[0] < importances[1] < importances[2] or importances[0] > importances[1] > importances[2]:
                plt.plot(list_of_scales, importances, label=city)
            # plt.scatter(list_of_scales, importances, label=city, style)

        plt.title(feature)
        plt.yscale("log")
        plt.xscale("log")
        plt.legend(ncol=3)
        plt.tight_layout()
        plt.savefig(MEAN_MAX + "_case_PDP/log_computed_FI_" + feature + ".png")


"""