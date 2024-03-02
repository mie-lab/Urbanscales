import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import sys

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from xgboost import XGBRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import config


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
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import make_scorer, explained_variance_score, r2_score
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

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




    # Example of how to call the function
    # compare_models_gof(X, Y, scaling=True)

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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from itertools import product
from sklearn.preprocessing import PolynomialFeatures

import pandas
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, explained_variance_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np

def compare_models_gof_standard_cv(X, Y, feature_list, cityname, scale, tod,  scaling=True, include_interactions=True):

    X = X[feature_list]

    if include_interactions:
        # Generate interaction terms (degree=2 includes original features, their squares, and interaction terms)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X = poly.fit_transform(X)
        feature_list = poly.get_feature_names_out(feature_list)  # Update feature_list with new polynomial feature names

    # Cross-validation strategy
    kf = KFold(n_splits=15, shuffle=True, random_state=42)

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, max_depth=200, min_samples_leaf=2),
        "Gradient Boosting Machine": GradientBoostingRegressor(n_estimators=100, random_state=42),
        # "Gradient Boosting Machine": XGBRegressor(n_estimators=100, random_state=42),
        "Lasso": Lasso(random_state=42),
        "Ridge": Ridge(random_state=42)
    }

    # Results dictionary

    results_explained_variance = {}
    results_mse = {}
    models_trained = {}
    for name, model in models.items():
        if scaling:
            # If scaling is true, include a scaler in the pipeline
            pipeline = make_pipeline(StandardScaler(), model)
        else:
            pipeline = make_pipeline(model)

        # Convert X back to DataFrame if it was transformed to numpy array by PolynomialFeatures
        if isinstance(X, np.ndarray):
            X_df = pandas.DataFrame(X, columns=feature_list)
        else:
            X_df = X

        sprint (model, X_df.shape)
        # Perform cross-validation and store the mean explained variance score
        cv_results_explained_variance = cross_val_score(pipeline, X_df, Y, cv=kf,
                                                        scoring=make_scorer(explained_variance_score))
        cv_results_mse = cross_val_score(pipeline, X_df, Y, cv=kf, scoring=make_scorer(mean_squared_error))
        results_explained_variance[name] = np.mean(cv_results_explained_variance)
        results_mse[name] = np.mean(cv_results_mse)

        # Train the model on the full dataset and store it
        pipeline.fit(X_df, Y)
        models_trained[name] = pipeline

    # Print the results
    print("Model Performance Comparison (Explained variance):")
    print("-------------------------------------------------------------")

    for name, score in results_explained_variance.items():
        print(f"{name}: {score:.4f}")

    print("Model Performance Comparison (MSE):")
    print("-------------------------------------------------------------")

    for name, score in results_mse.items():
        print(f"{name}: {score:.4f}")

    import shap


    from sklearn.model_selection import cross_val_predict
    rf_model = models_trained["Random Forest"].named_steps['randomforestregressor']
    if config.MASTER_VISUALISE_EACH_STEP:
        for i, tree in enumerate(rf_model.estimators_):
            print(f"Tree {i + 1}:")
            print(f"  Depth: {tree.tree_.max_depth}")
            print(f"  Number of leaves: {tree.tree_.n_leaves}")
            print(f"  Number of features: {tree.tree_.n_features}")
            print("---")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(StandardScaler().fit_transform(X))

    # Aggregate SHAP values
    shap_values_agg = shap_values

    # Plot feature importance
    total_shap_values = np.abs(shap_values_agg).mean(axis=0)
    # After computing total_shap_values
    total_shap_values = np.abs(shap_values_agg).mean(axis=0)

    # Create a list of tuples (feature name, total SHAP value)
    feature_importance_list = [(feature, shap_value) for feature, shap_value in zip(X.columns, total_shap_values)]

    # Sort the list based on SHAP values in descending order
    sorted_feature_importance_list = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)

    # Write the sorted list to a file
    output_file_path = os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
                                    f"tod_{tod}_total_feature_importance_scale_{scale}.csv")
    with open(output_file_path, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["Feature", "Total SHAP Value"])
        for feature, shap_value in sorted_feature_importance_list:
            csvwriter.writerow([feature, shap_value])

    # Compute Otsu threshold
    otsu_threshold = threshold_otsu(total_shap_values)
    plt.clf()
    shap.summary_plot(shap_values_agg, X, plot_type="bar", show=False)
    if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots")):
        os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots"))
    plt.title("Otsu: " + str(otsu_threshold))
    plt.tight_layout()
    plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots",
                             f"tod_{tod}_shap_total_FI_scale_{scale}_.png"))
    plt.clf()





    # Filter features based on Otsu threshold
    filtered_features = [idx for idx, val in enumerate(total_shap_values) if val > otsu_threshold]

    for idx in filtered_features:
        feature = X.columns[idx]
        # feature = feature_list[idx] # X.columns[idx]
        shap.dependence_plot(feature, shap_values_agg, X, show=False)
        if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots")):
            os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots"))
        plt.ylim(-1, 3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots", f"tod_{tod}_shap_pdp_{feature}_scale_{scale}.png"))
        plt.clf()
    # Plot SHAP-based PDP for filtered features
    if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "GOF" + f"tod_{tod}_scale_{scale}.csv")):
        with open(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "GOF" + f"tod_{tod}_scale_{scale}.csv"), "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(["model", "GoF_explained_Variance", "GoF_MSE", "TOD", "Scale", "cityname"])

    with open(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "GOF" + f"tod_{tod}_scale_{scale}.csv"), "a") as f:
        csvwriter = csv.writer(f)
        for name, score in results_explained_variance.items():
            print(f"{name}: {score:.4f}")
            csvwriter.writerow([name, results_explained_variance[name], results_mse[name], tod, scale, cityname])

def compare_models_gof_spatial_cv(X, Y, feature_list, bbox_to_strip, cityname, scaling=True, include_interactions=True, n_strips=3):
    # Use only the selected features
    X = X[feature_list]

    # If interaction terms should be included, generate them
    if include_interactions:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X = poly.fit_transform(X)
        feature_list = poly.get_feature_names_out(feature_list)  # Get new feature names

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=20),
        "Gradient Boosting Machine": GradientBoostingRegressor(n_estimators=100, random_state=42),
        # "Gradient Boosting Machine": XGBRegressor(n_estimators=100, random_state=42),
        "Lasso": Lasso(random_state=42),
        "Ridge": Ridge(random_state=42)
    }

    # Results dictionary to store model scores
    results = {name: [] for name in models}
    models_trained = {}
    # Perform spatial cross-validation using the spatial splits
    for strip_index in range(n_strips):
        index_to_bbox = {i: tuple(bbox.keys())[0] for i, bbox in enumerate(bboxes)}

        test_mask = X.index.isin(
            [i for i, bbox_tuple in index_to_bbox.items() if bbox_to_strip[bbox_tuple] == strip_index])
        train_mask = ~test_mask

        # Split the data into training and test sets based on spatial split
        X_train, X_test = X[train_mask], X[test_mask]
        Y_train, Y_test = Y[train_mask], Y[test_mask]

        # If scaling is required, scale the features
        if scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        sprint (X_train.shape, X_test.shape, strip_index)

        # Train and evaluate each model
        for name, model in models.items():
            cloned_model = clone(model)
            cloned_model.fit(X_train, Y_train)
            score = explained_variance_score(Y_test, cloned_model.predict(X_test))
            results[name].append(score)
            models_trained[name] = cloned_model

    # Print the results
    for name, scores in results.items():
        print(f"{name}:")
        print(f"Scores for each fold: {scores}")
        print(f"Average score explained variance: {np.mean(scores)}\n")

    import shap
    from sklearn.model_selection import cross_val_predict
    rf_model = models_trained["Random Forest"]
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(StandardScaler().fit_transform(X))

    # Aggregate SHAP values
    shap_values_agg = shap_values

    # Plot feature importance
    total_shap_values = np.abs(shap_values_agg).mean(axis=0)

    # Create a list of tuples (feature name, total SHAP value)
    feature_importance_list = [(feature, shap_value) for feature, shap_value in zip(X.columns, total_shap_values)]

    # Sort the list based on SHAP values in descending order
    sorted_feature_importance_list = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)

    # Write the sorted list to a file
    if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "spatial")):
        os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname,"spatial"))
    output_file_path = os.path.join(config.BASE_FOLDER, config.network_folder, cityname,"spatial",
                                    f"spatial_total_feature_importance.csv")
    with open(output_file_path, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["Feature", "Total SHAP Value"])
        for feature, shap_value in sorted_feature_importance_list:
            csvwriter.writerow([feature, shap_value])

    plt.clf()
    shap.summary_plot(shap_values_agg, X, plot_type="bar", show=False)
    if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots_spatial")):
        os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots_spatial"))
    plt.title("Feature Importance (Spatial)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots_spatial",
                             f"spatial_shap_total_FI.png"))
    plt.clf()

    # Compute Otsu threshold
    otsu_threshold = threshold_otsu(total_shap_values)

    # Filter features based on Otsu threshold
    filtered_features = [idx for idx, val in enumerate(total_shap_values) if val > otsu_threshold]

    for idx in filtered_features:
        feature = X.columns[idx]
        shap.dependence_plot(feature, shap_values_agg, X, show=False)
        if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, "PDP_plots_spatial")):
            os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, "PDP_plots_spatial"))
        plt.ylim(-1, 3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, "PDP_plots_spatial", f"spatial_shap_pdp_{feature}.png"))
        plt.clf()


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import pandas as pd


def shuffle(X, Y, random_state=None):
    # Generate a permutation based on the length of Y
    np.random.seed(random_state)
    perm = np.random.permutation(len(Y))

    # If X is a DataFrame, use .iloc for indexing
    if isinstance(X, pd.DataFrame):
        X_shuffled = X.iloc[perm].reset_index(drop=True)
    # If X is a numpy array, use numpy indexing
    elif isinstance(X, np.ndarray):
        X_shuffled = X[perm]
    else:
        raise TypeError("X must be a pandas DataFrame or a numpy array.")

    # Apply the same for Y, checking if it's a pandas Series or numpy array
    if isinstance(Y, pd.Series):
        Y_shuffled = Y.iloc[perm].reset_index(drop=True)
    elif isinstance(Y, np.ndarray):
        Y_shuffled = Y[perm]
    else:
        raise TypeError("Y must be a pandas Series or a numpy array.")

    return X_shuffled, Y_shuffled


# Note: Ensure X is a DataFrame and Y is a Series or similar before calling this function

if __name__ == "__main__":
    list_of_cities = "Singapore|Zurich|Mumbai|Auckland|Istanbul|MexicoCity|Bogota|NewYorkCity|Capetown|London".split("|")
    list_of_cities_list_of_list = [
                                    # list_of_cities[:2],
                                    # list_of_cities[2:]
                                    # [list_of_cities[0]],
                                    # [list_of_cities[1]],
                                    # [list_of_cities[2]],
                                    # [list_of_cities[3]],
                                    # [list_of_cities[4]],
                                    # [list_of_cities[5]],
                                    # [list_of_cities[6]],
                                    # [list_of_cities[7]],
                                    # [list_of_cities[8]],
                                    # [list_of_cities[9]],
                                    list(config.rn_city_wise_bboxes.keys())
                                ]

    tod_list_of_list = config.ps_tod_list

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
        'streets_per_node_count_5',
        'total_crossings'
    ]

    scale_list = config.scl_list_of_seeds

    results = {}
    for list_of_cities in list_of_cities_list_of_list:
        for tod_list in tod_list_of_list:
            for scale in scale_list:
                for city in list_of_cities:

                    tod = tod_list
                    x = []
                    y = []
                    fname = os.path.join(config.BASE_FOLDER, config.network_folder, city, f"_scale_{scale}_train_data_{tod}.pkl")
                    try:
                        temp_obj = CustomUnpicklerTrainDataVectors(open(fname, "rb")).load()
                        if isinstance(temp_obj.X, pd.DataFrame):
                            filtered_X = temp_obj.X[list(common_features)]
                    except:
                        print ("Error in :")
                        sprint(list_of_cities, tod_list, scale, city)
                        continue


                    def calculate_strip_boundaries(bboxes, n_strips=7):
                        # Extract all the longitude values from the bounding boxes
                        all_lons = [list(bbox.keys())[0][3] for bbox in bboxes] + [list(bbox.keys())[0][2] for bbox
                                                                                   in bboxes]

                        # Sort the longitude values
                        sorted_lons = sorted(list(set(all_lons)))

                        # Calculate the number of bboxes per strip
                        bboxes_per_strip = len(sorted_lons) // n_strips

                        # Initialize the list of strip boundaries
                        strip_boundaries = [sorted_lons[0]]

                        # Determine the boundaries of each strip
                        for i in range(1, n_strips):
                            boundary_index = i * bboxes_per_strip
                            strip_boundaries.append(sorted_lons[boundary_index])

                        # Add the last longitude value as the end boundary of the last strip
                        strip_boundaries.append(sorted_lons[-1])

                        return strip_boundaries

                    def assign_bboxes_to_strips(bboxes, strip_boundaries):
                        strip_assignments = {}
                        bbox_to_strip = {}
                        for i, bbox in enumerate(bboxes):
                            bbox_coords = list(bbox.keys())[0]
                            West, East = bbox_coords[3], bbox_coords[
                                2]  # Assuming bbox_coords is in the format (North, South, East, West)


                            # Find the strip that contains the bbox
                            for strip_index in range(len(strip_boundaries) - 1):
                                left_boundary = strip_boundaries[strip_index]
                                right_boundary = strip_boundaries[strip_index + 1]

                                # Check if bbox is within the current strip's boundaries
                                if West >= left_boundary and East <= right_boundary:
                                    if strip_index not in strip_assignments:
                                        strip_assignments[strip_index] = []
                                    strip_assignments[strip_index].append(bbox_coords)
                                    bbox_to_strip[bbox_coords] = strip_index
                                    break

                        return strip_assignments, bbox_to_strip


                    from shapely.geometry import Polygon
                    def visualize_splits(bboxes, strip_assignments, strip_boundaries, bbox_to_strip,
                                         split_direction='vertical'):
                        # Create a GeoDataFrame for the bounding boxes
                        gdf = gpd.GeoDataFrame({
                            'geometry': [
                                Polygon([(West, North), (East, North), (East, South), (West, South)])
                                for bbox in bboxes
                                for North, South, East, West in [list(bbox.keys())[0]]
                            ],
                            'strip': [
                                bbox_to_strip[(North, South, East, West)]
                                for bbox in bboxes
                                for North, South, East, West in [list(bbox.keys())[0]]
                            ]
                        }, crs="EPSG:4326")

                        # Plot the bounding boxes with different colors for each strip
                        if config.MASTER_VISUALISE_EACH_STEP:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            for strip, color in zip(range(len(strip_boundaries) - 1),
                                                    ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'cyan']):
                                gdf[gdf['strip'] == strip].plot(ax=ax, color=color, alpha=0.5, edgecolor='black')

                            # Add split boundaries
                            if split_direction == 'vertical':
                                for lon in strip_boundaries:
                                    plt.axvline(x=lon, color='black', linestyle='--', linewidth=3)
                            elif split_direction == 'horizontal':
                                for lat in strip_boundaries:
                                    plt.axhline(y=lat, color='black', linestyle='--', linewidth=3)

                            plt.xlabel('Longitude')
                            plt.ylabel('Latitude')
                            plt.title('Spatial Splits')
                            plt.show()


                    bboxes = temp_obj.bbox_X
                    n_strips = 5

                    strip_boundaries = calculate_strip_boundaries(bboxes, n_strips)
                    strip_assignments, bbox_to_strip = assign_bboxes_to_strips(bboxes, strip_boundaries)
                    from shapely.geometry import Polygon

                    visualize_splits(bboxes, strip_assignments, strip_boundaries, bbox_to_strip,
                                     split_direction='vertical')
                    # After processing each city and time of day, concatenate data
                    x.append(temp_obj.X)
                    y.append(temp_obj.Y)

                    # Concatenate the list of DataFrames in x and y
                    X = pd.concat(x, ignore_index=True)
                    # Convert any NumPy arrays in the list to Pandas Series
                    y_series = [pd.Series(array) if isinstance(array, np.ndarray) else array for array in y]

                    # Concatenate the list of Series and DataFrames
                    Y = pd.concat(y_series, ignore_index=True)

                    sprint (city, scale, tod, config.shift_tile_marker, X.shape, Y.shape)

                    # for column in common_features:
                    #     # plt.hist(X[column], 50)
                    #     # plt.title("Histogram for " + column)
                    #     # plt.hist(X[column])
                    #     # plt.show()
                    #
                    #     plt.title("Histogram for Y")
                    #     plt.hist(Y)
                    #     plt.show()
                    #     break

                    # locs = (Y > 0.5)  # & (Y < 6 )
                    # X = X[locs]
                    # Y = Y[locs]

                    # compare_models_gof_standard_cv(X, Y, common_features, tod=tod, cityname=city, scale=scale, include_interactions=False, scaling=True)
                    compare_models_gof_spatial_cv(X, Y, common_features, include_interactions=False, scaling=True,
                                                  bbox_to_strip=bbox_to_strip, n_strips=n_strips, cityname=city)

                    if config.MASTER_VISUALISE_EACH_STEP:
                        # Plot the bboxes from scl_jf
                        # Example list of bounding boxes
                        for column in common_features:

                            bboxes = [list(i.keys())[0] for i in temp_obj.bbox_X]
                            from shapely.geometry import Polygon

                            values_list = temp_obj.X[column]

                            # Normalize the values for coloring
                            values_normalized = (values_list - np.min(values_list)) / (
                                        np.max(values_list) - np.min(values_list))

                            # Create a GeoDataFrame including the values for heatmap
                            try:
                                gdf = gpd.GeoDataFrame({
                                    'geometry': [Polygon([(lon1, lat1), (lon1, lat2), (lon2, lat2), (lon2, lat1)]) for
                                                 lat1, lat2, lon1, lon2 in bboxes],
                                    'value': values_normalized
                                }, crs="EPSG:4326")  # EPSG:4326 is WGS84 latitude-longitude projection
                            except:
                                gdf = gpd.GeoDataFrame({
                                    'geometry': [Polygon([(lon1, lat1), (lon1, lat2), (lon2, lat2), (lon2, lat1)])
                                                 for
                                                 lat1, lat2, lon1, lon2, _unused_len_ in bboxes],
                                    'value': values_normalized
                                }, crs="EPSG:4326")  # EPSG:4326 is WGS84 latitude-longitude projection

                            # Convert the GeoDataFrame to the Web Mercator projection
                            gdf_mercator = gdf.to_crs(epsg=3857)

                            # Plotting with heatmap based on values and making boundaries invisible
                            fig, ax = plt.subplots(figsize=(10, 10))
                            gdf_mercator.plot(ax=ax, column='value', cmap='viridis', edgecolor='none',
                                              alpha=0.7)  # Use 'viridis' or any other colormap
                            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                            ax.set_axis_off()
                            plt.title(column)
                            # plt.colorbar()
                            plt.show()

                        if config.MASTER_VISUALISE_EACH_STEP:
                            # Plot the bboxes from scl_jf
                            # Example list of bounding boxes
                            bboxes = [list(i.keys())[0] for i in temp_obj.bbox_X]
                            from shapely.geometry import Polygon

                            values_list = temp_obj.Y

                            # Normalize the values for coloring
                            values_normalized = (values_list - np.min(values_list)) / (
                                    np.max(values_list) - np.min(values_list))

                            try:
                                gdf = gpd.GeoDataFrame({
                                    'geometry': [Polygon([(lon1, lat1), (lon1, lat2), (lon2, lat2), (lon2, lat1)]) for
                                                 lat1, lat2, lon1, lon2 in bboxes],
                                    'value': values_normalized
                                }, crs="EPSG:4326")  # EPSG:4326 is WGS84 latitude-longitude projection
                            except:
                                gdf = gpd.GeoDataFrame({
                                    'geometry': [Polygon([(lon1, lat1), (lon1, lat2), (lon2, lat2), (lon2, lat1)])
                                                 for
                                                 lat1, lat2, lon1, lon2, _unused_len_ in bboxes],
                                    'value': values_normalized
                                }, crs="EPSG:4326")  # EPSG:4326 is WGS84 latitude-longitude projection

                            # Convert the GeoDataFrame to the Web Mercator projection
                            gdf_mercator = gdf.to_crs(epsg=3857)

                            # Plotting with heatmap based on values and making boundaries invisible
                            fig, ax = plt.subplots(figsize=(10, 10))
                            gdf_mercator.plot(ax=ax, column='value', cmap='viridis', edgecolor='none',
                                              alpha=0.7)  # Use 'viridis' or any other colormap
                            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                            ax.set_axis_off()
                            plt.title("Y")
                            # plt.colorbar()
                            plt.show()
                        # input("Enter any key to continue for different TOD")


