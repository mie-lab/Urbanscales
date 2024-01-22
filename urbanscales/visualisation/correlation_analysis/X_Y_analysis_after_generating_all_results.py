import os
import sys

import shap

sys.path.append("../../../")
import pickle
import copy
from sklearn.model_selection import KFold, train_test_split
from smartprint import smartprint as sprint
from scipy import stats

current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.join(current_dir, '..')
os.chdir(current_dir)
sprint (os.getcwd())
import geopandas as gpd
import contextily as ctx
from shapely.geometry import box
import geopy.distance as gpy_dist

# for the custom unpickler
from urbanscales.preprocessing.train_data import TrainDataVectors


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
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
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




#
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import contextily as ctx
# from shapely.geometry import box
# import geopy.distance as gpy_dist
# import numpy as np

def plot_bboxes_for_debugging(temp_obj, identifier):
    bbox_list = temp_obj.bbox_X  # Assuming this is the correct attribute

    # Create a list of box geometries
    geometries = []
    area_list = []
    x_distance_list = []
    y_distance_list = []
    for bbox in bbox_list:
        # Assuming bbox is structured as (N, S, E, W, id)
        N, S, E, W, _ = list(bbox.keys())[0]
        geometries.append(box(W, S, E, N))

        x_distance = gpy_dist.geodesic((N, E), (N, W)).km
        y_distance = gpy_dist.geodesic((N, E), (S, E)).km
        area = x_distance * y_distance

        area_list.append(area)
        x_distance_list.append(x_distance)
        y_distance_list.append(y_distance)


    # Create GeoDataFrame from the geometries
    gdf = gpd.GeoDataFrame({'geometry': geometries}, crs='EPSG:4326')

    # Convert to Web Mercator for basemap
    gdf = gdf.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(15, 10))
    gdf.boundary.plot(ax=ax, color='blue', linewidth=2)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    plt.savefig("Bboxes_final_After_feature_Extraction_" + identifier + ".png", dpi=300)
    plt.show(block=False)



    mean_area = np.mean(area)
    median_area = np.median(area)
    mean_x = np.mean(x_distance)
    mean_y = np.mean(y_distance)
    median_x = np.median(x_distance)
    median_y = np.median(y_distance)
    with open("Mean_area_of_tiles.txt", "a") as f:
        f.write(f"{identifier} Mean Geodesic Area: {mean_area} km^2; {mean_x} km; {mean_y} km\n")
        f.write(f"{identifier} Median Geodesic Area: {median_area} km^2; {median_x} km; {median_y} km\n")

# Example usage
# plot_bboxes_for_debugging(temp_obj, "_identifier_")


# Example usage
# plot_bboxes_for_debugging(temp_obj, "_identifier_")

import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold


import seaborn as sns
import matplotlib.pyplot as plt

def KDE_plots(X, y, identifier):
    # Assuming `X` is your features DataFrame and `y` is your target variable
    for feature_name in X.columns: # Replace with your feature name
        plt.clf()
        # Create a DataFrame for plotting
        plot_data = X.copy()
        plot_data['target'] = y

        # Create the KDE plot
        sns.kdeplot(data=plot_data, x=feature_name, y='target')
        plt.title(f'KDE of {feature_name} and target variable')
        plt.xlabel(feature_name)
        plt.ylabel('Jam factor')
        plt.title(identifier)
        plt.savefig("SHAP_plots/KDE_" + identifier + "_" + feature_name + ".png")
        plt.show(block=False)

def compare_models(X, y, identifier):
    # Assuming X and y are your features and target
    n_splits = 7
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=30, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators = 30, random_state=42)
    }

    # Function to compute cross-validated explained variance and R-squared
    def get_cv_scores(model, X, y, cv):
        scores_explained_variance = cross_val_score(model, X, y, cv=cv, scoring='explained_variance')
        scores_r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
        scores_mse =  cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        return np.mean(scores_explained_variance), np.std(scores_explained_variance), np.mean(scores_r2), np.std(
            scores_r2), np.mean(scores_mse), np.std(scores_mse)

    sprint (X.shape, y.shape)
    # Compute and print the cross-validated scores for each model
    for name, model in models.items():
        mean_ev, std_ev, mean_r2, std_r2, mean_mse, std_mse = get_cv_scores(model, X, y, kf)
        print(
            "{:60}, {:<30}  Mean MSE = {:.4f}, Std Dev MSE = {:.4f}".format(
                f"{name}:", identifier, mean_mse, std_mse)) # Mean Explained Variance = {:.4f}, Std Dev EV = {:.4f}, # mean_r2, std_r2,


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence
from sklearn.base import BaseEstimator, RegressorMixin

class PerfectModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.y_ = y  # Store the target values
        self.is_fitted_ = True  # Add an attribute to indicate the model has been fitted
        return self

    def predict(self, X):
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise Exception("This PerfectModel instance is not fitted yet.")
        return self.y_

def plot_PDPs(X, y, identifier):
    """
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest and Linear Regression models
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    # gbm_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    lr_model = LinearRegression()
    perfect_model = PerfectModel()

    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    # gbm_model.fit(X_train, y_train)
    perfect_model.fit(X_train, y_train)  # This will store y_train in the model

    # "Perfect Model" - a hypothetical model where predictions exactly match the targets
    perfect_model_predictions = y_train  # Assuming perfect predictions are equal to y_train

    # Plot PDPs for all models
    for feature in range(X.shape[1]):
        plt.figure(figsize=(8, 5))

        # Random Forest PDP
        pdp_rf = partial_dependence(rf_model, X_train, features=[feature], grid_resolution=20)
        plt.plot(pdp_rf['grid_values'][0], pdp_rf.average[0], label=f'RF: PDP for {X.columns[feature]}', color='tab:orange')

        # pdp_gbm = partial_dependence(gbm_model, X_train, features=[feature], grid_resolution=20)
        # plt.plot(pdp_rf['grid_values'][0], pdp_gbm.average[0], label=f'GBM: PDP for {X.columns[feature]}', color='tab:red')

        # Linear Regression PDP
        pdp_lr = partial_dependence(lr_model, X_train, features=[feature], grid_resolution=20)
        plt.plot(pdp_lr['grid_values'][0], pdp_lr.average[0], label=f'LR: PDP for {X.columns[feature]}', color='tab:blue')

        # # Perfect Model PDP
        # pdp_perfect = partial_dependence(perfect_model, X_train, features=[feature], grid_resolution=20)
        # plt.plot(pdp_perfect['grid_values'][0], pdp_perfect.average[0], label=f'Perfect: PDP for {X.columns[feature]}',
        #          color='tab:green')

        # Plot settings
        plt.title(f"PDP for Feature: {X.columns[feature]}")
        plt.xlabel(X.columns[feature])
        plt.ylabel('Partial dependence')
        plt.legend()
        plt.savefig(f"SHAP_plots/PDP_{identifier}_{X.columns[feature]}.png")
        plt.show(block=False)
        plt.clf()  # Clear figure for next plot
    """

    # Initialize KFold
    n_splits = 7
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    lr_model = LinearRegression()
    lr_model.fit(X, y)

    # Function to calculate average PDP across folds
    def calculate_average_pdp(model, X, y, feature, grid_resolution=20):
        pdp_values_sum = None
        for train_index, test_index in kf.split(X):
            X_train, y_train = X.iloc[train_index], y[train_index]
            model.fit(X_train, y_train)
            pdp_result = partial_dependence(model, X_train, features=[feature], grid_resolution=grid_resolution)

            if pdp_values_sum is None:
                pdp_values_sum = pdp_result.average[0]
            else:
                pdp_values_sum += pdp_result.average[0]

        # Average the PDP values
        pdp_values_avg = pdp_values_sum / n_splits
        return pdp_result['grid_values'][0], pdp_values_avg

    # Example usage
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)

    plt.clf()

    for feature in range(X.shape[1]):
        # plt.clf()
        if identifier.split("_")[0] == "Istanbul" and feature == "global_betweenness":
            continue
        try:
            grid_values, pdp_avg = calculate_average_pdp(rf_model, X, y, feature)
        except ValueError:
            sprint ("Error in: ", identifier, feature)
            continue

        plt.plot(grid_values, pdp_avg, label=f'RF: PDP for {X.columns[feature]}')
        # plt.plot(grid_values, pdp_avg, label=f'RF: PDP for {X.columns[feature]}', color='tab:orange')
        pdp_lr = partial_dependence(lr_model, X, features=[feature], grid_resolution=20)
        # plt.plot(pdp_lr['grid_values'][0], pdp_lr.average[0], label=f'LR: PDP for {X.columns[feature]}',
        #          color='tab:blue')
        # plt.title(identifier + f"PDP for Feature: {X.columns[feature]}")
        plt.title(identifier)
        # plt.xlabel(X.columns[feature])
        plt.xlabel("Feature")
        # plt.ylabel('Partial dependence')
        plt.ylabel('Jam Factor')

    plt.legend()
    plt.ylim(0,8)
    plt.tight_layout()
    # plt.savefig(f"SHAP_plots/PDP_{identifier}_{X.columns[feature]}.png")
    plt.savefig(f"SHAP_plots/PDP_{identifier}.png")
    plt.show(block=False)


def plot_SHAP_PDP(X, y, identifier):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Create the SHAP explainer and calculate SHAP values for the test set
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Plot SHAP values for each feature across all data points
    for feature in range(X.shape[1]):
        # Create a new figure for each plot
        plt.figure()
        shap.dependence_plot(ind=feature, shap_values=shap_values, features=X_test, show=False)

        # Save the plot before showing it
        filename = f"SHAP_plots/SHAP_PDP_{identifier}_{X.columns[feature]}.png"
        plt.savefig(filename, bbox_inches='tight')

        # Show the plot
        plt.show(block=False)

        # Clear the current figure after saving and showing
        plt.clf()


# Example usage with your data


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from scipy.cluster.hierarchy import dendrogram, linkage

if __name__ == "__main__":
    os.system("rm Mean_area_of_tiles.txt")
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
        'total_crossings'
    ]

    scale_list = [25, 50, 100] # 25, 50, 100]

    results = {}

    for scale in scale_list:
        all_cities_data = []

        for city in list_of_cities:
            combined_X_for_city = []

            for tod in range(6, 7):
                fname = f"/Users/nishant/Documents/GitHub/WCS/network_tmax_smean_50x50_Jan_19/{city}/_scale_{scale}_train_data_{tod}.pkl"
                try:
                    temp_obj = CustomUnpicklerTrainDataVectors(open(fname, "rb")).load()

                    if isinstance(temp_obj.X, pd.DataFrame):
                        try:
                            filtered_X = temp_obj.X[list(common_features)]
                            combined_X_for_city.append(filtered_X)

                        except KeyError as e:

                            filtered_X = temp_obj.X[list (set(common_features) - set(["global_betweenness"]) )]
                            combined_X_for_city.append(filtered_X)

                            print ("\n\n")
                            print ("Some features not found:", e)
                            sprint(city, scale, tod, temp_obj.X.shape, temp_obj.Y.shape)
                            print ("Proceeding without this feature")
                            print("\n\n")
                            debug_pitstop = True
                            # raise Exception("Exiting execution; feature not found")
                            # continue
                            do_nothing = True
                            # combined_X_for_city.append("Dummy")


                except FileNotFoundError:
                    print("Error in fname:", fname)

            assert len(combined_X_for_city) == 1 # since now we are only using a single tod; no TOD combination implies
                                                 # only a single value will be present


            debug_pitstop = True




            X = temp_obj.X #[common_features]
            X = X.drop(columns=[col for col in X if col not in common_features])

            Y = temp_obj.Y

            areas = {
                'betweenness': 1,
                'circuity_avg': 1,
                'global_betweenness': 1 ,# /( (scale/50)**2 ),
                'k_avg': 1,
                'lane_density': 1,
                'm': 1/( (scale/50)**2 ),
                'n': 1/( (scale/50)**2 ),
                'metered_count': 1/( (scale/50)**2 ),
                'non_metered_count': 1/( (scale/50)**2 ),
                'street_length_total': 1, # 1/( (scale/50)**2 ),
                # 'streets_per_node_count_5': 1/( (scale/50)**2 ),
                'total_crossings':  1/( (scale/50)**2 ),

            }

            for column in X.columns:
                X[column] = X[column] / areas[column]

            # sprint(city, scale, tod, X.shape, Y.shape)

            debug_pitstop = True

            # plot_SHAP_PDP(X, Y, identifier=city + "_scl_" + str(scale) + "_tod_" + str(tod))
            plot_PDPs(X, Y, identifier=city + "_scl_" + str(scale) + "_tod_" + str(tod))
            # compare_models(X, Y, identifier=city + "_scl_" + str(scale) + "_tod_" + str(tod))
            # KDE_plots(X, Y, identifier=city + "_scl_" + str(scale) + "_tod_" + str(tod))
            print ("\n\n")
            # a = 1/0