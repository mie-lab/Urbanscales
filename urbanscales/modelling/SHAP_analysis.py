import csv
import os
import time

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
config.MASTER_VISUALISE_EACH_STEP = True


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

from sklearn.metrics import make_scorer, explained_variance_score as explained_variance_scorer, r2_score
import geopandas as gpd
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

    # def random_forest_feature_importance(self, common_features):
    #     n_splits = 7
    #
    #     kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    #     total_feature_importances = np.zeros(len(common_features))
    #
    #     for train_index, test_index in kfold.split(self.nparrayX, self.nparrayY):
    #         X_train, X_test = self.nparrayX[train_index], self.nparrayX[test_index]
    #         Y_train, Y_test = self.nparrayY[train_index], self.nparrayY[test_index]
    #         rf = RandomForestRegressor(n_estimators=100, random_state=42)
    #         rf.fit(X_train, Y_train)
    #         total_feature_importances += rf.feature_importances_
    #     total_feature_importances /= n_splits
    #
    #     return total_feature_importances

    # def plot_feature_importances(self, importances, scale, common_features, list_of_cities):
    #     plt.title(f"Feature Importances RF " + list_of_cities)
    #     plt.plot(range(len(common_features)), importances, marker='o', linestyle='-', label=f"Scale: {scale}")
    #     plt.xticks(range(len(common_features)), common_features, rotation=90)
    #     plt.xlabel('Feature')
    #     plt.ylabel('Importance')
    #     plt.legend()


import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.base import clone

def compare_models_gof_standard_cv(X, Y, feature_list, cityname, scale, tod,  n_splits,scaling=True, include_interactions=True):

    X = X[feature_list]

    if include_interactions:
        # Generate interaction terms (degree=2 includes original features, their squares, and interaction terms)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X = poly.fit_transform(X)
        feature_list = poly.get_feature_names_out(feature_list)  # Update feature_list with new polynomial feature names

    # Cross-validation strategy
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42), # , max_depth=200, min_samples_leaf=2),
        # "Gradient Boosting Machine": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting Machine": XGBRegressor(n_estimators=200, random_state=42),
        "Lasso": Lasso(random_state=42),
        "Ridge": Ridge(random_state=42)
    }

    # Results dictionary

    results_explained_variance = {}
    results_mse = {}
    models_trained = {name: [] for name in models}
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

        split_counter = -1
        for train_index, test_index in kf.split(X_df):
            split_counter += 1
            X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

            # Clone the model and fit it on the training data
            cloned_model = clone(pipeline)
            cloned_model.fit(X_train, Y_train)

            output_file_path = os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
                                            f"tod_{tod}_GOF_NON_SPATIAL_CV_SHAPE_of_dataframe_scale_{scale}_.csv")
            with open(output_file_path, "a") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow([cityname, scale, config.CONGESTION_TYPE, tod, "X_train.shape", X_train.shape, "X_test.shape", X_test.shape, "split_index", split_counter ])


            # Evaluate the model on the test data
            explained_variance_score = explained_variance_scorer(Y_test, cloned_model.predict(X_test))
            mse = mean_squared_error(Y_test, cloned_model.predict(X_test))

            # Store the results and the trained model
            results_explained_variance.setdefault(name, []).append(explained_variance_score)
            results_mse.setdefault(name, []).append(mse)
            models_trained[name].append(cloned_model)



    output_file_path = os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
                                    f"tod_{tod}_GOF_CV_MEAN_AND_STD_scale_{scale}.csv")
    with open(output_file_path, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["Model", "Explained-var-Mean-across-CV", "Explained-var-STD-across-CV",
                            "MSE-Mean-across-CV", "MSE-STD-across-CV"] + ["Explained-var-CV-values"+ str(x)
                                                                          for x in range(1, len(results_explained_variance[name]) + 1)] +
                           ["MSE-var-CV-values"+ str(x) for x in range(1, len(results_mse[name]) + 1)])
        for name in models:
            csvwriter.writerow([name, np.mean(results_explained_variance[name]), np.std(results_explained_variance[name]),
                                np.mean(results_mse[name]), np.std(results_mse[name])] +
                               results_explained_variance[name] + results_mse[name])

    # Compute the mean scores for each model
    for name in models:
        results_explained_variance[name] = np.mean(results_explained_variance[name])
        results_mse[name] = np.mean(results_mse[name])

    # Print the results
    print("\n\n-------------------------------------------------------------")
    print("Non Spatial CV Model Performance Comparison (Explained variance):")

    for name, score in results_explained_variance.items():
        print(f"{name}: {score:.4f}")

    print("\n\n-------------------------------------------------------------")
    print("Non Spatial CV Model Performance Comparison (MSE):")

    for name, score in results_mse.items():
        print(f"{name}: {score:.4f}")



    # Initialize a list to store SHAP values for each fold
    shap_values_list = []
    import shap


    # Perform cross-validation
    split_counter = -1
    for train_index, test_index in kf.split(X_df):
        split_counter += 1
        X_train, X_test = pd.DataFrame(X_df.iloc[train_index]), pd.DataFrame(X_df.iloc[test_index])
        Y_train, Y_test = pd.DataFrame(Y.iloc[train_index]), pd.DataFrame(Y.iloc[test_index])


        if X_train.shape[0] < 5 or X_test.shape[0] < 5:
            print("Skipped the strip since very few Test or train data in the strip, split_counter=", split_counter)
            sprint(X_train.shape, X_test.shape)
            continue

        # If scaling is required, scale the features
        if scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_whole_data_standardised = scaler.transform(X)




        # Compute SHAP values for the trained model and append to the list
        rf_model = models_trained["Random Forest"][split_counter].named_steps['randomforestregressor']
        import shap
        print ("Starting explainer: ")
        starttime = time.time()
        explainer = shap.TreeExplainer(rf_model)
        print ("Explainer completed in ", time.time() - starttime, "seconds")
        # shap_values = explainer.shap_values(X_test)  # # computing shap for test data no post processing needed for statndardisation since the X_test is already standardised

        starttime = time.time()
        shap_values = explainer.shap_values(
            X_whole_data_standardised)  # # computing shap for test data no post processing needed for statndardisation since the X_test is already standardised
        shap_values_list.append(shap_values)
        print ("explainer.shap_values() completed in ", time.time() - starttime, "seconds")

    # Average the SHAP values across all folds
    # concatenated_shap_values = np.concatenate(shap_values_list, axis=0)

    shap_values_stack = np.stack(shap_values_list, axis=0)

    # Compute the mean SHAP values across the first axis (the CV splits axis)
    mean_shap_values = np.mean(shap_values_stack, axis=0)

    # Aggregate SHAP values
    shap_values_agg = mean_shap_values

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
    plt.clf(); plt.close()
    shap.summary_plot(shap_values_agg, X, plot_type="bar", show=False)
    if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots")):
        os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots"))
    plt.title("Otsu: " + str(otsu_threshold))
    plt.tight_layout()
    plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots",
                             f"tod_{tod}_shap_total_FI_scale_{scale}_.png"))
    plt.clf(); plt.close()





    # Filter features based on Otsu threshold
    # filtered_features = [idx for idx, val in enumerate(total_shap_values) if val > otsu_threshold]
    filtered_features = [idx for idx, val in enumerate(total_shap_values) if val > 0]  # Removed this filter to plot all cases

    for idx in filtered_features:
        feature = X.columns[idx]
        # feature = feature_list[idx] # X.columns[idx]
        shap.dependence_plot(feature, shap_values_agg, X, show=False)
        if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots")):
            os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots"))
        plt.ylim(-1, 3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots", f"tod_{tod}_shap_pdp_{feature}_scale_{scale}.png"))
        plt.clf(); plt.close()
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

def compare_models_gof_spatial_cv(X, Y, feature_list, bbox_to_strip, cityname, tod, scale, temp_obj, scaling=True, include_interactions=True, n_strips=3):
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
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42), # , max_depth=20),
        # "Gradient Boosting Machine": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Machine": XGBRegressor(n_estimators=200, random_state=42),
        "Lasso": Lasso(random_state=42),
        "Ridge": Ridge(random_state=42)
    }

    # Results dictionary to store model scores
    results_explained_variance = {name: [] for name in models}
    results_mse = {name: [] for name in models}

    models_trained = {}

    bboxes = temp_obj.bbox_X
    gdf = gpd.GeoDataFrame({
        'geometry': [
            Polygon([(West, North), (East, North), (East, South), (West, South)])
            for bbox in bboxes
            for North, South, East, West in [list(bbox.keys())[0]]
        ],
        'index': range(len(bboxes))
    }, crs="EPSG:4326")


    # Perform spatial cross-validation using the spatial splits
    for strip_index in range(n_strips):
        a = []
        for i in range(len(temp_obj.bbox_X)):
            if bbox_to_strip[list(temp_obj.bbox_X[i].keys())[0]] == strip_index:
                a.append(i)

        test_mask = X.index.isin(a)
        train_mask = ~test_mask

        # Split the data into training and test sets based on spatial split
        X_train, X_test = pd.DataFrame(X[train_mask]), pd.DataFrame(X[test_mask])
        Y_train, Y_test = pd.DataFrame(Y[train_mask]), pd.DataFrame(Y[test_mask])
        if X_train.shape[0] < 5 or X_test.shape[0] < 5:
            print ("Skipped the strip since very few Test or train data in the strip, strip_index=", strip_index)
            sprint (X_train.shape, X_test.shape)
            continue

        # Plot train and test sets
        if config.MASTER_VISUALISE_EACH_STEP:
            fig, ax = plt.subplots(figsize=(10, 6))
            gdf_train = gdf[gdf['index'].isin(X_train.index)]
            gdf_test = gdf[gdf['index'].isin(X_test.index)]
            gdf_train.plot(ax=ax, color='green', alpha=0.5, edgecolor='black', label='Train Set')
            gdf_test.plot(ax=ax, color='blue', alpha=0.5, edgecolor='black', label='Test Set')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('Spatial Split: Train and Test Sets')
            plt.legend()
            plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "Spatial_split_train_test_during_model_training" + str(scale)  + "_split_" +str(strip_index) + ".png"), dpi=300)
            plt.show(block=False); plt.close()

        # If scaling is required, scale the features
        if scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Train and evaluate each model
        for name, model in models.items():
            cloned_model = clone(model)
            cloned_model.fit(X_train, Y_train)
            explained_variance_score = explained_variance_scorer(Y_test, cloned_model.predict(X_test))
            mse = mean_squared_error(Y_test, cloned_model.predict(X_test))
            results_explained_variance[name].append(explained_variance_score)
            results_mse[name].append(mse)
            if name in models_trained:
                models_trained[name].append (cloned_model)
            else:
                models_trained[name] = [cloned_model]


            output_file_path = os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
                                            f"tod_{tod}_GOF_SPATIAL_CV_SHAPE_of_dataframe_scale_{scale}_.csv")
            with open(output_file_path, "a") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow([cityname, scale, config.CONGESTION_TYPE, tod, "X_train.shape", X_train.shape, "X_test.shape", X_test.shape, "split_index", strip_index ])

    shap_values_list = []



    output_file_path = os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
                                    f"tod_{tod}_GOF_SPATIAL_CV_MEAN_AND_STD_scale_{scale}.csv")
    with open(output_file_path, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["Model", "Explained-var-Mean-across-CV", "Explained-var-STD-across-CV",
                            "MSE-Mean-across-CV", "MSE-STD-across-CV"] + ["Explained-var-CV-values"+ str(x)
                                                                          for x in range(1, len(results_explained_variance[name]) + 1)] +
                           ["MSE-var-CV-values"+ str(x) for x in range(1, len(results_mse[name]) + 1)])
        for name in models:
            csvwriter.writerow([name, np.mean(results_explained_variance[name]), np.std(results_explained_variance[name]),
                                np.mean(results_mse[name]), np.std(results_mse[name])] +
                               results_explained_variance[name] + results_mse[name])

    for name in results_explained_variance:
        print("Before computing mean: ", name)
        print(results_explained_variance[name])
        results_explained_variance[name] = np.mean(results_explained_variance[name])
        results_mse[name] = np.mean(results_mse[name])

    print("\n\n-------------------------------------------------------------")
    print("results_explained_variance spatial")
    for name, scores in results_explained_variance.items():
        print(f"{name}:")
        print(f"Scores for each fold: {scores}")
        print(f"Average score explained variance: {np.mean(scores)}\n")

    # Print the results
    print("\n\n-------------------------------------------------------------")
    print("results_MSE spatial")
    for name, scores in results_mse.items():
        print(f"{name}:")
        print(f"Scores for each fold: {scores}")
        print(f"Average score MSE: {np.mean(scores)}\n")

    for strip_index in range(n_strips):
        a = []
        for i in range(len(temp_obj.bbox_X)):
            if bbox_to_strip[list(temp_obj.bbox_X[i].keys())[0]] == strip_index:
                a.append(i)

        test_mask = X.index.isin(a)
        train_mask = ~test_mask

        # Split the data into training and test sets based on spatial split
        X_train, X_test = X[train_mask], X[test_mask]


        if X_train.shape[0] < 5 or X_test.shape[0] < 5:
            print("Skipped the strip since very few Test or train data in the strip, strip_index=", strip_index)
            sprint(X_train.shape, X_test.shape)
            continue

        # If scaling is required, scale the features
        if scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_whole_data_standardised = scaler.transform(X)


        rf_model = models_trained["Random Forest"][strip_index]
        import shap
        explainer = shap.TreeExplainer(rf_model)
        # shap_values = explainer.shap_values(X_test)  # # computing shap for test data no post processing needed for statndardisation since the X_test is already standardised
        shap_values = explainer.shap_values(X_whole_data_standardised)  # # computing shap for test data no post processing needed for statndardisation since the X_test is already standardised
        shap_values_list.append(shap_values)



    # Average the SHAP values across all folds
    # concatenated_shap_values = np.concatenate(shap_values_list, axis=0)

    shap_values_stack = np.stack(shap_values_list, axis=0)

    # Compute the mean SHAP values across the first axis (the CV splits axis)
    mean_shap_values = np.mean(shap_values_stack, axis=0)

    # Aggregate SHAP values
    shap_values_agg = mean_shap_values
    # Aggregate SHAP values
    shap_values_agg = mean_shap_values

    # Plot feature importance
    total_shap_values = np.abs(shap_values_agg).mean(axis=0)

    # Create a list of tuples (feature name, total SHAP value)
    feature_importance_list = [(feature, shap_value) for feature, shap_value in zip(X.columns, total_shap_values)]

    # Sort the list based on SHAP values in descending order
    sorted_feature_importance_list = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)

    # Write the sorted list to a file
    spatial_folder = os.path.join(config.BASE_FOLDER, config.network_folder, cityname, f"spatial_tod_{tod}_scale_{scale}")
    if not os.path.exists(spatial_folder):
        os.makedirs(spatial_folder)
    output_file_path = os.path.join(spatial_folder, "spatial_total_feature_importance.csv")
    with open(output_file_path, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["Feature", "Total SHAP Value"])
        for feature, shap_value in sorted_feature_importance_list:
            csvwriter.writerow([feature, shap_value])

    plt.clf(); plt.close()
    shap.summary_plot(shap_values_agg, X, plot_type="bar", show=False)
    plt.title("Feature Importance (Spatial)")
    plt.tight_layout()
    plt.savefig(os.path.join(spatial_folder, "spatial_shap_total_FI.png"))
    plt.clf(); plt.close()

    # Compute Otsu threshold
    otsu_threshold = threshold_otsu(total_shap_values)

    # Filter features based on Otsu threshold
    # filtered_features = [idx for idx, val in enumerate(total_shap_values) if val > otsu_threshold]
    filtered_features = [idx for idx, val in enumerate(total_shap_values) if val > 0]  # Removed this filter to plot all cases


    for idx in filtered_features:
        feature = X.columns[idx]
        shap.dependence_plot(feature, shap_values_agg, X, show=False)
        plt.ylim(-1, 3)
        plt.tight_layout()
        plt.savefig(os.path.join(spatial_folder, f"spatial_shap_pdp_{feature}.png"))
        plt.clf(); plt.close()

    # Write GOF results to a file
    gof_file_path = os.path.join(spatial_folder, "GOF_spatial_tod_"+ str(tod) + "_scale_"+ str(scale) + ".csv")
    if not os.path.exists(gof_file_path):
        with open(gof_file_path, "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(["model", "GoF_explained_Variance", "GoF_MSE", "TOD", "Scale", "cityname"])

    with open(gof_file_path, "a") as f:
        csvwriter = csv.writer(f)
        for name, score in results_explained_variance.items():
            csvwriter.writerow([name, score, results_mse[name], tod, scale, cityname])



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
        # 'streets_per_node_count_5',
        'total_crossings'
    ]

    scale_list = config.scl_list_of_seeds

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



                    def calculate_grid_boundaries(bboxes, n_strips=7):
                        # Extract all the longitude and latitude values from the bounding boxes
                        all_lons = [bbox_coords[3] for bbox in bboxes for bbox_coords in bbox.keys()] + \
                                   [bbox_coords[2] for bbox in bboxes for bbox_coords in bbox.keys()]
                        all_lats = [bbox_coords[0] for bbox in bboxes for bbox_coords in bbox.keys()] + \
                                   [bbox_coords[1] for bbox in bboxes for bbox_coords in bbox.keys()]

                        # Sort the longitude and latitude values
                        sorted_lons = sorted(list(set(all_lons)))
                        sorted_lats = sorted(list(set(all_lats)))

                        # Calculate the number of bboxes per strip for both longitude and latitude
                        lons_per_strip = len(sorted_lons) // n_strips
                        lats_per_strip = len(sorted_lats) // n_strips

                        # Initialize the list of grid boundaries for longitude and latitude
                        lon_boundaries = [sorted_lons[0]]
                        lat_boundaries = [sorted_lats[0]]

                        # Determine the boundaries of each strip for longitude
                        for i in range(1, n_strips):
                            lon_boundary_index = i * lons_per_strip
                            lon_boundaries.append(sorted_lons[lon_boundary_index])

                        # Add the last longitude value as the end boundary of the last strip
                        lon_boundaries.append(sorted_lons[-1])

                        # Determine the boundaries of each strip for latitude
                        for i in range(1, n_strips):
                            lat_boundary_index = i * lats_per_strip
                            lat_boundaries.append(sorted_lats[lat_boundary_index])

                        # Add the last latitude value as the end boundary of the last strip
                        lat_boundaries.append(sorted_lats[-1])

                        return lon_boundaries, lat_boundaries


                    def assign_bboxes_to_grid(bboxes, lon_boundaries, lat_boundaries, n_strips):
                        grid_assignments = {}
                        bbox_to_grid = {}
                        for i, bbox in enumerate(bboxes):
                            bbox_coords = list(bbox.keys())[0]
                            North, South, East, West = bbox_coords

                            # Find the grid cell that contains the bbox
                            for lon_index in range(len(lon_boundaries) - 1):
                                for lat_index in range(len(lat_boundaries) - 1):
                                    left_boundary = lon_boundaries[lon_index]
                                    right_boundary = lon_boundaries[lon_index + 1]
                                    bottom_boundary = lat_boundaries[lat_index]
                                    top_boundary = lat_boundaries[lat_index + 1]

                                    # Check if bbox is within the current grid cell's boundaries
                                    if West >= left_boundary and East <= right_boundary and North <= top_boundary and South >= bottom_boundary:
                                        grid_index = lon_index * n_strips + lat_index # (lon_index, lat_index)
                                        if grid_index not in grid_assignments:
                                            grid_assignments[grid_index] = []
                                        grid_assignments[grid_index].append(bbox_coords)
                                        bbox_to_grid[bbox_coords] = grid_index
                                        break

                        return grid_assignments, bbox_to_grid


                    from shapely.geometry import Polygon


                    def visualize_splits(bboxes, strip_boundaries, bbox_to_strip,
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

                            import geopy.distance

                            # Calculate the average latitude of your bounding boxes
                            a = []
                            latlist = []
                            lonlist = []
                            for i in range(len(bboxes)):
                                a.append( np.mean ( [list(bboxes[i].keys())[0][0], list(bboxes[i].keys())[0][1]]  ))
                                latlist.append(list(bboxes[i].keys())[0][0])
                                latlist.append(list(bboxes[i].keys())[0][1])  # The order is NSEW
                                lonlist.append(list(bboxes[i].keys())[0][2])
                                lonlist.append(list(bboxes[i].keys())[0][3])  # The order is NSEW


                            average_latitude = np.mean(a)
                            sprint (average_latitude)

                            km_in_degrees_lon = \
                            geopy.distance.distance(kilometers=1).destination((average_latitude, 0), bearing=90)[1]
                            km_in_degrees_lat = \
                            geopy.distance.distance(kilometers=1).destination((average_latitude, 0), bearing=0)[
                                0] - average_latitude

                            # Choose a corner for the scale box (e.g., bottom right)
                            corner_lon = max(lonlist)
                            corner_lat = min(latlist)

                            # Create the scale box
                            scale_box = Polygon([
                                (corner_lon - km_in_degrees_lon, corner_lat),
                                (corner_lon, corner_lat),
                                (corner_lon, corner_lat + km_in_degrees_lat),
                                (corner_lon - km_in_degrees_lon, corner_lat + km_in_degrees_lat),
                                (corner_lon - km_in_degrees_lon, corner_lat)
                            ])

                            # Add the scale box to the plot
                            ax.plot(*scale_box.exterior.xy, color='tab:blue')

                            # Add labels for the 1 km edges
                            ax.text(corner_lon - km_in_degrees_lon / 2, corner_lat - 0.0005, '1 km', ha='center',
                                    fontsize=4)
                            ax.text(corner_lon + 0.0005, corner_lat + km_in_degrees_lat / 2, '1 km', va='center',
                                    rotation='vertical', fontsize=4)

                            # Save and show the plot
                            plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, city,
                                                     "Spatial_splits_for_spatial_CV" + str(scale) + ".png"), dpi=300)
                            plt.show(block=False); plt.close()



                    def visualize_grid(bboxes, lon_boundaries, lat_boundaries, bbox_to_grid, split_direction='grid'):
                        # Create a GeoDataFrame for the bounding boxes
                        gdf = gpd.GeoDataFrame({
                            'geometry': [
                                Polygon([(West, North), (East, North), (East, South), (West, South)])
                                for bbox in bboxes
                                for North, South, East, West in [list(bbox.keys())[0]]
                            ],
                            'grid_cell': [
                                bbox_to_grid[list(bbox.keys())[0]]
                                for bbox in bboxes
                            ]
                        }, crs="EPSG:4326")

                        # Plot the bounding boxes with different colors for each grid cell
                        if config.MASTER_VISUALISE_EACH_STEP:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            unique_cells = set(gdf['grid_cell'])
                            colors = plt.cm.get_cmap('tab20', len(unique_cells))

                            for i, cell in enumerate(unique_cells):
                                gdf[gdf['grid_cell'] == cell].plot(ax=ax, color=colors(i), alpha=0.5, edgecolor='black')

                            # Add split boundaries
                            if split_direction == 'grid':
                                for lon in lon_boundaries:
                                    plt.axvline(x=lon, color='black', linestyle='--', linewidth=1)
                                for lat in lat_boundaries:
                                    plt.axhline(y=lat, color='black', linestyle='--', linewidth=1)

                            plt.xlabel('Longitude')
                            plt.ylabel('Latitude')
                            plt.title('Spatial Grid Splits')

                            # Save and show the plot
                            plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, city,
                                                     "Spatial_grid_splits_for_spatial_CV.png"), dpi=300)
                            plt.show(block=False); plt.close()


                    bboxes = temp_obj.bbox_X
                    n_strips = 2

                    from shapely.geometry import Polygon

                    if config.SHAP_mode_spatial_CV == "vertical":
                        n_strips = 7
                        strip_boundaries = calculate_strip_boundaries(bboxes, n_strips)
                        strip_assignments, bbox_to_strip = assign_bboxes_to_strips(bboxes, strip_boundaries)
                        visualize_splits(bboxes, strip_boundaries, bbox_to_strip,
                                         split_direction='vertical')
                    elif config.SHAP_mode_spatial_CV == "grid":
                        lon_boundaries, lat_boundaries = calculate_grid_boundaries(bboxes, n_strips=n_strips)
                        grid_assignments, bbox_to_strip = assign_bboxes_to_grid(bboxes, lon_boundaries, lat_boundaries, n_strips)
                        visualize_grid(bboxes, lon_boundaries, lat_boundaries, bbox_to_strip, split_direction='grid')
                    else:
                        raise Exception("Wrong split type; must be grid or vertical")


                    # After processing each city and time of day, concatenate data
                    x.append(temp_obj.X)
                    y.append(temp_obj.Y)

                    # Concatenate the list of DataFrames in x and y
                    assert len(x) == 1
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
                    #     # plt.show(block=False); plt.close()
                    #
                    #     plt.title("Histogram for Y")
                    #     plt.hist(Y)
                    #     plt.show(block=False); plt.close()
                    #     break

                    # locs = (Y > 0.5)  # & (Y < 6 )
                    # X = X[locs]
                    # Y = Y[locs]

                    if config.SHAP_mode_spatial_CV == "grid":
                        N_STRIPS = n_strips ** 2
                    elif config.SHAP_mode_spatial_CV == "vertical":
                        N_STRIPS = n_strips
                    else:
                        raise Exception("Wrong split type; must be grid or vertical")

                    model_fit_time_start = time.time()
                    compare_models_gof_standard_cv(X, Y, common_features, tod=tod, cityname=city, scale=scale, n_splits=7, include_interactions=False, scaling=True)
                    t_non_spatial = time.time() - model_fit_time_start
                    # compare_models_gof_spatial_cv(X, Y, common_features, temp_obj=temp_obj, include_interactions=False, scaling=True,
                    #                               bbox_to_strip=bbox_to_strip, n_strips=N_STRIPS, tod=tod, cityname=city, scale=scale)
                    # t_spatial = time.time() - model_fit_time_start - t_non_spatial
                    # sprint (t_spatial, t_non_spatial, "seconds")


                    if 2==3 and config.MASTER_VISUALISE_EACH_STEP:
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
                            plt.show(block=False); plt.close()

                        if 2==3 and config.MASTER_VISUALISE_EACH_STEP:
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
                            plt.show(block=False); plt.close()
                        # input("Enter any key to continue for different TOD")


