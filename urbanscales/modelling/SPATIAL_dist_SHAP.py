import csv
import os
import sys
import time

import contextily as ctx
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import shap
from skimage.filters import threshold_otsu
from skimage.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.metrics import explained_variance_score as explained_variance_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
import joblib


sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import config

# config.MASTER_VISUALISE_EACH_STEP = True

from sklearn.metrics import r2_score as r2_scorer

sys.path.append("../../../")
from urbanscales.preprocessing.train_data import TrainDataVectors

import pickle
import copy
from sklearn.model_selection import KFold

from smartprint import smartprint as sprint

current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.join(current_dir, '..')
os.chdir(current_dir)
sprint(os.getcwd())
from slugify import slugify


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


def compare_models_gof_standard_cv(X, Y, feature_list, cityname, scale, tod, n_splits,
                                   scaling=True, include_interactions=True):
    """
    Compares different regression models on their goodness-of-fit (GoF) using standard cross-validation.

    Parameters:
        X (DataFrame): The input features.
        Y (DataFrame): The output/target variable.
        feature_list (list): List of features to be used in the model.
        cityname (str): Name of the city for which the model is being evaluated.
        scale (int): Scale parameter that may represent a model-specific or data-specific scale.
        tod (str): Time of day or other temporal identifier.
        n_splits (int): Number of splits for cross-validation.
        scaling (bool, optional): If True, apply feature scaling. Defaults to True.
        include_interactions (bool, optional): If True, includes polynomial interaction terms. Defaults to True.

    Returns:
        None: The function internally prints out the performance metrics and may save them to files.
    """

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
        "LR": LinearRegression(),
        "RF": RandomForestRegressor(n_estimators=200, random_state=42),  # , max_depth=200, min_samples_leaf=2),
        # "Gradient Boosting Machine": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "GBM": XGBRegressor(n_estimators=200, random_state=42),
        "LLR": Lasso(random_state=42),
        "RLR": Ridge(random_state=42)
    }

    if config.SHAP_additive_regression_model:
        models = {
            "LR": LinearRegression(),
            "RF": RandomForestRegressor(n_estimators=200, random_state=42),
            # , max_depth=200, min_samples_leaf=2),
            # "Gradient Boosting Machine": GradientBoostingRegressor(n_estimators=200, random_state=42),
            "GBM": XGBRegressor(n_estimators=200, random_state=42, max_depth=1),
            "LLR": Lasso(random_state=42),
            "RLR": Ridge(random_state=42)
        }

    # Results dictionary

    results_explained_variance = {}
    results_mse = {}
    results_r2_Score = {}
    models_trained = {name: [] for name in models}
    for name, model in models.items():

        if scaling:
            # If scaling is true, include a Standardscaler in the pipeline
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
            sprint(name, X_train.shape)  # , train_index, test_index)

            if name == "RF":
                estimators = cloned_model.named_steps['randomforestregressor'].estimators_
                T = len(estimators)
                L = max([estimator.get_n_leaves() for estimator in estimators])
                D = max([estimator.tree_.max_depth for estimator in estimators])
                print(f"{name}: T={T}, L={L}, D={D}")
                sprint(cityname, scale, T * L * D)

            output_file_path = os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
                                            f"tod_{tod}_GOF_NON_SPATIAL_CV_SHAPE_of_dataframe_scale_{scale}_.csv")
            with open(output_file_path, "a") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(
                    [cityname, scale, config.CONGESTION_TYPE, tod, "X_train.shape", X_train.shape, "X_test.shape",
                     X_test.shape, "split_index", split_counter])

            # Evaluate the model on the test data
            explained_variance_score = explained_variance_scorer(Y_test, cloned_model.predict(X_test))
            mse = mean_squared_error(Y_test, cloned_model.predict(X_test))
            r2_score = r2_scorer(Y_test, cloned_model.predict(X_test))

            # Store the results and the trained model
            results_explained_variance.setdefault(name, []).append(explained_variance_score)
            results_mse.setdefault(name, []).append(mse)
            results_r2_Score.setdefault(name, []).append(r2_score)

            models_trained[name].append(cloned_model)

    output_file_path = os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
                                    f"tod_{tod}_GOF_CV_MEAN_AND_STD_scale_{scale}.csv")
    with open(output_file_path, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["Model", "Explained-var-Mean-across-CV", "Explained-var-STD-across-CV",
                            "MSE-Mean-across-CV", "MSE-STD-across-CV"] + ["Explained-var-CV-values" + str(x)
                                                                          for x in range(1, len(
                results_explained_variance[name]) + 1)] +
                           ["MSE-var-CV-values" + str(x) for x in range(1, len(results_mse[name]) + 1)])
        for name in models:
            csvwriter.writerow(
                [name, np.mean(results_explained_variance[name]), np.std(results_explained_variance[name]),
                 np.mean(results_mse[name]), np.std(results_mse[name])] +
                results_explained_variance[name] + results_mse[name])

    for name in models:
        if name == "RF":
            print("logging_for_NON-SPATIAL_explained_var", cityname, scale, np.mean(results_explained_variance[name]),
                  np.median(results_explained_variance[name]), "STANDARD_CV", config.CONGESTION_TYPE, sep=",")

    # Compute the mean scores for each model
    for name in models:
        results_explained_variance[name] = np.mean(results_explained_variance[name])
        results_mse[name] = np.mean(results_mse[name])
        results_r2_Score[name] = np.mean(results_r2_Score[name])

    # Print the results
    # print("\n\n-------------------------------------------------------------")
    # print("Non Spatial CV Model Performance Comparison (Explained variance):")

    for name, score in results_explained_variance.items():
        print(city, scale, name, score, "XpVar")

    # print("\n\n-------------------------------------------------------------")
    # print("Non Spatial CV Model Performance Comparison (MSE):")

    for name, score in results_mse.items():
        print(city, scale, name, score, "MSE")

    # print("\n\n-------------------------------------------------------------")
    # print("Non Spatial CV Model Performance Comparison Coefficient of Variation (R2):")

    for name, score in results_r2_Score.items():
        print(city, scale, name, score, "R2")

    if not config.SHAP_values_disabled:
        # Initialize a list to store SHAP values for each fold
        shap_values_list = []
        import shap

        # Perform cross-validation
        split_counter = -1
        for train_index, test_index in kf.split(X_df):
            split_counter += 1
            X_train, X_test = pd.DataFrame(X_df.iloc[train_index]), pd.DataFrame(X_df.iloc[test_index])
            Y_train, Y_test = pd.DataFrame(Y.iloc[train_index]), pd.DataFrame(Y.iloc[test_index])

            try:
                if X_train.shape[0] < 5 or X_test.shape[0] < 5:
                    print("Skipped the strip since very few Test or train data in the strip, split_counter=",
                          split_counter)
                    sprint(X_train.shape, X_test.shape)
                    continue
            except:
                # to skip when empty
                print("Skipped the strip since very few Test or train data in the strip, split_counter=", split_counter)
                continue

            # If scaling is required, scale the features
            if scaling:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                X_whole_data_standardised = scaler.transform(X)

            else:
                X_whole_data_standardised = X

            # Compute SHAP values for the trained model and append to the list
            if config.SHAP_additive_regression_model:
                rf_model = models_trained["Gradient Boosting Machine"][split_counter].named_steps['xgbregressor']
            else:
                rf_model = models_trained["Random Forest"][split_counter].named_steps['randomforestregressor']
            import shap
            print("Starting explainer: ")
            starttime = time.time()
            explainer = shap.TreeExplainer(rf_model)
            print("Explainer completed in ", time.time() - starttime, "seconds")
            # shap_values = explainer.shap_values(X_test)  # # computing shap for test data no post processing needed for statndardisation since the X_test is already standardised

            starttime = time.time()
            shap_values = explainer.shap_values(
                X_whole_data_standardised)  # # computing shap for test data no post processing needed for statndardisation since the X_test is already standardised
            shap_values_list.append(shap_values)
            print("explainer.shap_values() completed in ", time.time() - starttime, "seconds")
            if config.FAST_GEN_PDPs_for_multiple_runs and split_counter > 2:
                break  # just a single run for SHAP values for fast prototyping

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

        if config.SHAP_sort_features_alphabetical_For_heatmaps:
            feature_order = np.argsort(X.columns)
            X = X.iloc[:, feature_order]
            shap_values_agg = shap_values_agg[:, feature_order]

        otsu_threshold = threshold_otsu(total_shap_values)
        plt.clf();
        plt.close()
        shap.plots.bar(explainer(pd.DataFrame(X_whole_data_standardised, columns=X.columns)), show=False)
        if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots")):
            os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots"))
        plt.title("Otsu: " + str(otsu_threshold))
        plt.tight_layout()
        plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots",
                                 f"tod_{tod}_shap_total_FI_scale_{scale}_samething_.png"))
        plt.clf();
        plt.close()

        otsu_threshold = threshold_otsu(total_shap_values)
        plt.clf();
        plt.close()
        shap.summary_plot(shap_values_agg, X, plot_type="bar", show=False)
        if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots")):
            os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots"))
        plt.title("Otsu: " + str(otsu_threshold))
        plt.tight_layout()
        plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots",
                                 f"tod_{tod}_shap_total_FI_scale_{scale}_.png"))
        plt.clf();
        plt.close()

        plt.clf();
        plt.close()
        shap.plots.beeswarm(explainer(pd.DataFrame(X_whole_data_standardised, columns=X.columns)), show=False)
        if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots")):
            os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots"))
        plt.title("Otsu: " + str(otsu_threshold))
        plt.tight_layout()
        plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots",
                                 f"tod_{tod}_shap_total_FI_beeswarm_{scale}_.png"))
        plt.clf();
        plt.close()

        plt.clf();
        plt.close()
        shap.plots.heatmap(explainer(pd.DataFrame(X_whole_data_standardised, columns=X.columns)), show=False)
        if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots")):
            os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots"))
        plt.title("Otsu: " + str(otsu_threshold))
        plt.tight_layout()
        plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots",
                                 f"tod_{tod}_shap_total_FI_heatmap_{scale}_.png"))
        plt.clf();
        plt.close()

        # Filter features based on Otsu threshold
        # filtered_features = [idx for idx, val in enumerate(total_shap_values) if val > otsu_threshold]
        filtered_features = [idx for idx, val in enumerate(total_shap_values) if
                             val > 0]  # Removed this filter to plot all cases

        for idx in filtered_features:
            feature = X.columns[idx]
            # feature = feature_list[idx] # X.columns[idx]
            shap.dependence_plot(feature, shap_values_agg, X, show=False)
            if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots")):
                os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots"))
            plt.ylim(-1, 3)
            plt.tight_layout()
            plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots",
                                     f"tod_{tod}_shap_pdp_{feature}_scale_{scale}.png"))
            plt.clf();
            plt.close()
        # Plot SHAP-based PDP for filtered features

        if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
                                           "GOF" + f"tod_{tod}_scale_{scale}.csv")):
            with open(os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
                                   "GOF" + f"tod_{tod}_scale_{scale}.csv"), "w") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(["model", "GoF_explained_Variance", "GoF_MSE", "TOD", "Scale", "cityname"])

        with open(os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
                               "GOF" + f"tod_{tod}_scale_{scale}.csv"), "a") as f:
            csvwriter = csv.writer(f)
            for name, score in results_explained_variance.items():
                print(f"{name}: {score:.4f}")
                csvwriter.writerow([name, results_explained_variance[name], results_mse[name], tod, scale, cityname])

def plot_feature_influence(shap_values_df, feature_values_df, slugifiedstring):
    # Ensure the DataFrame is structured with SHAP values
    means = {}
    for column in shap_values_df.columns:
        # Identify positive and negative indices based on SHAP values
        pos_indices = shap_values_df[column] > 0
        neg_indices = shap_values_df[column] < 0

        # Calculate MeanSHAP for positive and negative indices
        MeanSHAP_pos = shap_values_df.loc[pos_indices, column].mean()
        MeanSHAP_neg = shap_values_df.loc[neg_indices, column].mean()

        # Calculate MeanX using the corresponding feature values
        MeanX_pos = feature_values_df.loc[pos_indices, column].mean()
        MeanX_neg = feature_values_df.loc[neg_indices, column].mean()

        # Calculate ExtendedDirection
        if pd.isna(MeanX_pos) or pd.isna(MeanX_neg):  # Handle cases where all SHAP are positive or negative
            ExtendedDirection = 0
            numerator = 0
            denominator = 0
        else:
            numerator = (MeanSHAP_pos - MeanSHAP_neg)
            denominator = (MeanX_pos - MeanX_neg)
            ExtendedDirection = numerator / denominator

        means[column] = (numerator, denominator, ExtendedDirection)

    # Plotting
    # fig, ax = plt.subplots()
    for i, (key, value) in enumerate(means.items()):
        # ax.arrow(i, 0, 0, value * 0.5, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        # ax.arrow(i, 0, 0, 1 * np.sign(value), head_width=0.1, head_length=0.1,width=abs(value) * 0.1, fc='blue', ec='blue')
        print(config.CONGESTION_TYPE, slugifiedstring, i, key, value[2], value[0], value[1])

    # ax.set_ylim(min(means.values()) - 1, max(means.values()) + 1)
    # plt.xticks(range(len(df.columns)), df.columns, rotation='vertical')
    # plt.xlabel("Features")
    # plt.ylabel("SHAP-Dir")
    # plt.show(block=False);
    # plt.close()




def compare_models_gof_standard_cv_HPT_new(X, Y, feature_list, cityname, scale, tod, n_splits,
                                           scaling=True, include_interactions=True):
    """
    Compares different regression models on their goodness-of-fit (GoF) using standard cross-validation
    with hyperparameter tuning for RF and GBM models.

    Parameters:
        X (DataFrame): The input features.
        Y (DataFrame): The output/target variable.
        feature_list (list): List of features to be used in the model.
        cityname (str): Name of the city for which the model is being evaluated.
        scale (int): Scale parameter that may represent a model-specific or data-specific scale.
        tod (str): Time of day or other temporal identifier.
        n_splits (int): Number of splits for cross-validation.
        scaling (bool, optional): If True, apply feature scaling. Defaults to True.
        include_interactions (bool, optional): If True, includes polynomial interaction terms. Defaults to True.

    Returns:
        None: The function internally prints out the performance metrics and may save them to files.
    """

    # Convert X back to DataFrame if it was transformed to numpy array by PolynomialFeatures
    if isinstance(X, np.ndarray):
        X_df = pandas.DataFrame(X, columns=feature_list)
    else:
        X_df = X

    sprint (X.shape, Y.shape, "Inside the model-fitting function")

    best_params_dict = {}
    # Filtering and potentially transforming the features
    X = X[feature_list]
    if include_interactions:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X = poly.fit_transform(X)
        feature_list = poly.get_feature_names_out()

    # Cross-validation strategy
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Define models
    models = {
        "LR": LinearRegression(),
        "RF": RandomForestRegressor(random_state=42),
        "GBM": XGBRegressor(random_state=42),
        "LLR": Lasso(random_state=42),
        "RLR": Ridge(random_state=42)
    }
    if config.SHAP_GBM_ENABLED:
        models = {
            "LR": LinearRegression(),
            "RF": RandomForestRegressor(random_state=42),
            "GBM": XGBRegressor(random_state=42),
            "LLR": Lasso(random_state=42),
            "RLR": Ridge(random_state=42)
        }
    else:
        models = {
            "LR": LinearRegression(),
            "RF": RandomForestRegressor(random_state=42),
            # "GBM": XGBRegressor(random_state=42),
            "LLR": Lasso(random_state=42),
            "RLR": Ridge(random_state=42)
        }

    rf_params = {
        'randomforestregressor__n_estimators': [50, 100, 200, 300, 400, 500, 600],  # Number of trees
        'randomforestregressor__max_depth': [10, 20, 50, 100],  # Maximum depth of each tree
        # 'randomforestregressor__min_samples_split': [2, 5, 10],
        # Minimum number of samples required to split an internal node
        # 'randomforestregressor__min_samples_leaf': [1, 2, 4, 5],  # Minimum number of samples required at a leaf node
        'randomforestregressor__max_features': ['sqrt'] # , 'log2']
        # Number of features to consider when looking for the best split
    }

    if config.SHAP_GBM_ENABLED:
        gbm_params = {
            'xgbregressor__n_estimators': [100, 200, 300, 400, 500, 1000],  # Number of boosting stages to perform
            'xgbregressor__learning_rate': [0.001, 0.01, 0.1, 0.2],  # Step size shrinkage used to prevent overfitting
            'xgbregressor__max_depth': [10, 20, 50, 100],  # Maximum depth of each tree
            # 'xgbregressor__min_samples_split': [2, 10],  # Minimum number of samples required to split an internal node
            # 'xgbregressor__subsample': [0.6, 0.8, 1.0]  # Subsample ratio of the training instances
        }
    else:
        gbm_params = {
            'xgbregressor__n_estimators': [2], #  [100, 200, 300, 400, 500, 1000],  # Number of boosting stages to perform
            'xgbregressor__learning_rate': [0.01],  # [0.001, 0.01, 0.1, 0.2],  # Step size shrinkage used to prevent overfitting
            'xgbregressor__max_depth': [2], #[10, 20, 50, 100],  # Maximum depth of each tree
            # 'xgbregressor__min_samples_split': [2, 10],  # Minimum number of samples required to split an internal node
            # 'xgbregressor__subsample': [0.6, 0.8, 1.0]  # Subsample ratio of the training instances
        }

    best_params = {}  # Store the best parameters

    # Evaluation metrics storage
    results_explained_variance = {}
    results_mse = {}
    results_r2_Score = {}

    for name, model in models.items():
        results_explained_variance[name] = []
        results_mse[name] = []
        results_r2_Score[name] = []

        if scaling:
            model = make_pipeline(StandardScaler(), model)
        else:
            model = make_pipeline(model)

        # Apply hyperparameter tuning for RF and GBM
        model_trials = clone(model)
        if name in ["RF", "GBM"]:
            param_grid = rf_params if name == "RF" else gbm_params
            if config.SHAP_random_search_CV:
                model_trials = RandomizedSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', refit=True,
                                                  n_iter=config.SHAP_HPT_num_iters, verbose=0)
            elif config.SHAP_Grid_search_CV:
                model_trials = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', refit=True, verbose=0)

            model_trials.fit(X, Y)

        # for train_index, test_index in kf.split(X):
        #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        #     Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        # Fit model

        # Save best parameters after fitting
        if name in ["RF", "GBM"]:  # and RandomizedSearchCV in type(model).__bases__:
            best_params[name] = model_trials.best_params_
            sprint(model_trials.best_params_)
            best_params_dict.update(
                model_trials.best_params_)  # since the params are different for GBM and RF (we keep the prefixes for each model type)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

            model_best = clone(model)  # for LR, RLR, LLR cases
            if name == "RF":
                model_best = RandomForestRegressor(
                    random_state=42,
                    n_estimators=best_params_dict["randomforestregressor__n_estimators"],
                    max_depth=best_params_dict["randomforestregressor__max_depth"],
                    max_features=best_params_dict["randomforestregressor__max_features"]
                )
                rf_model_best = clone(model_best)




            elif name == "GBM":
                model_best = XGBRegressor(
                    random_state=42,
                    max_depth=best_params_dict["xgbregressor__max_depth"],
                    n_estimators=best_params_dict["xgbregressor__n_estimators"],
                    learning_rate=best_params_dict["xgbregressor__learning_rate"],
                )
            elif name == "LR":
                model_best = LinearRegression()

            if not os.path.exists(os.path.join(config.BASE_FOLDER,  "results")):
                os.mkdir(os.path.join(config.BASE_FOLDER,  "results"))

            model_best.fit(X_train, Y_train)
            joblib.dump(model_best, os.path.join(config.BASE_FOLDER,  "results", "RF-" + slugify(str((config.CONGESTION_TYPE, cityname, scale, tod))) + ".joblib"))

            # if we wish to looad: just use this
            # loaded_rf = joblib.load("my_random_forest.joblib")

            y_pred = model_best.predict(X_test)
            explained_variance = explained_variance_score(Y_test, y_pred)
            mse = mean_squared_error(Y_test, y_pred)
            r2 = r2_score(Y_test, y_pred)

            results_explained_variance[name].append(explained_variance)
            results_mse[name].append(mse)
            results_r2_Score[name].append(r2)


    if not config.SHAP_values_disabled:
        if scaling:
            X = StandardScaler().fit_transform(X)
        rf_model_best.fit(X, Y)


        explainer = shap.TreeExplainer(rf_model_best)



        if config.SHAP_generate_beeswarm:
            plt.clf()
            dfX = pd.DataFrame(X, columns=feature_list)

            dfX = dfX.rename(columns={
                    'n': '#nodes',
                    'm': '#edges',
                    'k_avg': 'avg degree',
                    'streets_per_node_avg': 'SPN-avg',
                    'circuity_avg': 'avg circuity',
                    'metered_count': '#traffic lights',
                    'non_metered_count': '#free turns',
                    'total_crossings': '#total crossings',
                    'betweenness': 'local centrality',
                    'mean_lanes': 'avg lanes',
                    'streets_per_node_count_1': '#SPN-1',
                    'streets_per_node_count_2': '#SPN-2',
                    'streets_per_node_count_3': '#SPN-3',
                    'streets_per_node_count_4': '#SPN-4',
                    'streets_per_node_count_5': '#SPN-5',
                    'global_betweenness': 'global centrality'
                })


            shap_values = explainer(dfX)
            shap.plots.beeswarm(shap_values, show=False)
            plt.tight_layout()
            plt.savefig("SHAP_Demo" +slugify(str((config.CONGESTION_TYPE, cityname, scale, tod)))+ ".pdf", dpi=300)
            plt.show(block=False)


        # shap.summary_plot(shap_values, X, plot_type="bar")
        shap_values_without_columns = explainer.shap_values(X)
        shap_values_df = pd.DataFrame(shap_values_without_columns, columns=feature_list)

        plot_feature_influence(shap_values_df, pd.DataFrame(X, columns=feature_list),
                               slugifiedstring=slugify(str((config.CONGESTION_TYPE, cityname, scale, tod))))

        print("SHAP Values for each feature:")
        # print(slugify(str((config.CONGESTION_TYPE, cityname, scale, tod))), shap_values_df.abs().mean(axis=0))


        mean_abs_shap = shap_values_df.abs().mean(axis=0)
        print(f"{'Slug':<30} {'Feature Name':<20} {'Mean Abs SHAP Value':>15}")
        slug = slugify(str((config.CONGESTION_TYPE, cityname, scale, tod)))
        for feature, value in mean_abs_shap.iteritems():
            print(f"SHAP-- {slug:<30} {feature:<20} {value:>15.4f}")


        if not os.path.exists(os.path.join(config.BASE_FOLDER, "results")):
            os.mkdir(os.path.join(config.BASE_FOLDER, "results"))
        shap_values_df.mean(axis=0).to_csv(
            os.path.join(config.BASE_FOLDER, "results", slugify(str((config.CONGESTION_TYPE, cityname, scale, tod))) + "shap_values.csv"))


    for name, score in results_explained_variance.items():
        print(config.CONGESTION_TYPE, city, scale, name, np.mean(score), "XpVar")

    # print("\n\n-------------------------------------------------------------")
    # print("Non Spatial CV Model Performance Comparison (MSE):")

    for name, score in results_mse.items():
        print(config.CONGESTION_TYPE, city, scale, name, np.mean(score), "MSE")

    # print("\n\n-------------------------------------------------------------")
    # print("Non Spatial CV Model Performance Comparison Coefficient of Variation (R2):")

    for name, score in results_r2_Score.items():
        print(config.CONGESTION_TYPE, city, scale, name, np.mean(score), "R2")

    # if not config.SHAP_values_disabled:
    #
    #     # Initialize a list to store SHAP values for each fold
    #     shap_values_list = []
    #     import shap
    #
    #     # Perform cross-validation
    #     split_counter = -1
    #     for train_index, test_index in kf.split(X_df):
    #         split_counter += 1
    #         X_train, X_test = pd.DataFrame(X_df.iloc[train_index]), pd.DataFrame(X_df.iloc[test_index])
    #         Y_train, Y_test = pd.DataFrame(Y.iloc[train_index]), pd.DataFrame(Y.iloc[test_index])
    #
    #         try:
    #             if X_train.shape[0] < 5 or X_test.shape[0] < 5:
    #                 print("Skipped the strip since very few Test or train data in the strip, split_counter=",
    #                       split_counter)
    #                 sprint(X_train.shape, X_test.shape)
    #                 continue
    #         except:
    #             # to skip when empty
    #             print("Skipped the strip since very few Test or train data in the strip, split_counter=", split_counter)
    #             continue
    #
    #         # If scaling is required, scale the features
    #         if scaling:
    #             scaler = StandardScaler()
    #             X_train = scaler.fit_transform(X_train)
    #             X_test = scaler.transform(X_test)
    #             X_whole_data_standardised = scaler.transform(X)
    #
    #         else:
    #             X_whole_data_standardised = X
    #
    #         # Compute SHAP values for the trained model and append to the list
    #         if config.SHAP_additive_regression_model:
    #             rf_model = models_trained["Gradient Boosting Machine"][split_counter].named_steps['xgbregressor']
    #         else:
    #             rf_model = models_trained["Random Forest"][split_counter].named_steps['randomforestregressor']
    #         import shap
    #         print("Starting explainer: ")
    #         starttime = time.time()
    #         explainer = shap.TreeExplainer(rf_model)
    #         print("Explainer completed in ", time.time() - starttime, "seconds")
    #         # shap_values = explainer.shap_values(X_test)  # # computing shap for test data no post processing needed for statndardisation since the X_test is already standardised
    #
    #         starttime = time.time()
    #         shap_values = explainer.shap_values(
    #             X_whole_data_standardised)  # # computing shap for test data no post processing needed for statndardisation since the X_test is already standardised
    #         shap_values_list.append(shap_values)
    #         print("explainer.shap_values() completed in ", time.time() - starttime, "seconds")
    #         if config.FAST_GEN_PDPs_for_multiple_runs and split_counter > 2:
    #             break  # just a single run for SHAP values for fast prototyping
    #
    #     # Average the SHAP values across all folds
    #     # concatenated_shap_values = np.concatenate(shap_values_list, axis=0)
    #
    #     shap_values_stack = np.stack(shap_values_list, axis=0)
    #
    #     # Compute the mean SHAP values across the first axis (the CV splits axis)
    #     mean_shap_values = np.mean(shap_values_stack, axis=0)
    #
    #     # Aggregate SHAP values
    #     shap_values_agg = mean_shap_values
    #
    #     # After computing total_shap_values
    #     total_shap_values = np.abs(shap_values_agg).mean(axis=0)
    #
    #     # Create a list of tuples (feature name, total SHAP value)
    #     feature_importance_list = [(feature, shap_value) for feature, shap_value in zip(X.columns, total_shap_values)]
    #
    #     # Sort the list based on SHAP values in descending order
    #     sorted_feature_importance_list = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)
    #
    #     # Write the sorted list to a file
    #     output_file_path = os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
    #                                     f"tod_{tod}_total_feature_importance_scale_{scale}.csv")
    #     with open(output_file_path, "w") as f:
    #         csvwriter = csv.writer(f)
    #         csvwriter.writerow(["Feature", "Total SHAP Value"])
    #         for feature, shap_value in sorted_feature_importance_list:
    #             csvwriter.writerow([feature, shap_value])
    #
    #     # Compute Otsu threshold
    #
    #     if config.SHAP_sort_features_alphabetical_For_heatmaps:
    #         feature_order = np.argsort(X.columns)
    #         X = X.iloc[:, feature_order]
    #         shap_values_agg = shap_values_agg[:, feature_order]
    #
    #     otsu_threshold = threshold_otsu(total_shap_values)
    #     plt.clf();
    #     plt.close()
    #     shap.plots.bar(explainer(pd.DataFrame(X_whole_data_standardised, columns=X.columns)), show=False)
    #     if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots")):
    #         os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots"))
    #     plt.title("Otsu: " + str(otsu_threshold))
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots",
    #                              f"tod_{tod}_shap_total_FI_scale_{scale}_samething_.png"))
    #     plt.clf();
    #     plt.close()
    #
    #     otsu_threshold = threshold_otsu(total_shap_values)
    #     plt.clf();
    #     plt.close()
    #     shap.summary_plot(shap_values_agg, X, plot_type="bar", show=False)
    #     if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots")):
    #         os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots"))
    #     plt.title("Otsu: " + str(otsu_threshold))
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots",
    #                              f"tod_{tod}_shap_total_FI_scale_{scale}_.png"))
    #     plt.clf();
    #     plt.close()
    #
    #     plt.clf();
    #     plt.close()
    #     shap.plots.beeswarm(explainer(pd.DataFrame(X_whole_data_standardised, columns=X.columns)), show=False)
    #     if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots")):
    #         os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots"))
    #     plt.title("Otsu: " + str(otsu_threshold))
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots",
    #                              f"tod_{tod}_shap_total_FI_beeswarm_{scale}_.png"))
    #     plt.clf();
    #     plt.close()
    #
    #     plt.clf();
    #     plt.close()
    #     shap.plots.heatmap(explainer(pd.DataFrame(X_whole_data_standardised, columns=X.columns)), show=False)
    #     if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots")):
    #         os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots"))
    #     plt.title("Otsu: " + str(otsu_threshold))
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots",
    #                              f"tod_{tod}_shap_total_FI_heatmap_{scale}_.png"))
    #     plt.clf();
    #     plt.close()
    #
    #     # Filter features based on Otsu threshold
    #     # filtered_features = [idx for idx, val in enumerate(total_shap_values) if val > otsu_threshold]
    #     filtered_features = [idx for idx, val in enumerate(total_shap_values) if
    #                          val > 0]  # Removed this filter to plot all cases
    #
    #     for idx in filtered_features:
    #         feature = X.columns[idx]
    #         # feature = feature_list[idx] # X.columns[idx]
    #         shap.dependence_plot(feature, shap_values_agg, X, show=False)
    #         if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots")):
    #             os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots"))
    #         plt.ylim(-1, 3)
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "PDP_plots",
    #                                  f"tod_{tod}_shap_pdp_{feature}_scale_{scale}.png"))
    #         plt.clf();
    #         plt.close()
    #     # Plot SHAP-based PDP for filtered features
    #
    #     if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
    #                                        "GOF" + f"tod_{tod}_scale_{scale}.csv")):
    #         with open(os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
    #                                "GOF" + f"tod_{tod}_scale_{scale}.csv"), "w") as f:
    #             csvwriter = csv.writer(f)
    #             csvwriter.writerow(["model", "GoF_explained_Variance", "GoF_MSE", "TOD", "Scale", "cityname"])
    #
    #     with open(os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
    #                            "GOF" + f"tod_{tod}_scale_{scale}.csv"), "a") as f:
    #         csvwriter = csv.writer(f)
    #         for name, score in results_explained_variance.items():
    #             print(f"{name}: {score:.4f}")
    #             csvwriter.writerow([name, results_explained_variance[name], results_mse[name], tod, scale, cityname])




def compare_models_gof_spatial_cv(X, Y, feature_list, bbox_to_strip, cityname, tod, scale, temp_obj,
                                  scaling=True, include_interactions=True, n_strips=3):
    """
    Compares different regression models on their goodness-of-fit (GoF) using spatial cross-validation.

    Parameters:
        X (DataFrame): The input features.
        Y (DataFrame): The output/target variable.
        feature_list (list): List of features to be used in the model.
        bbox_to_strip (dict): Mapping of bounding box coordinates to spatial strips.
        cityname (str): Name of the city for which the model is being evaluated.
        tod (str): Time of day or other temporal identifier.
        scale (int): Scale parameter that may represent a model-specific or data-specific scale.
        temp_obj (object): An object that includes bbox_X which helps in spatial partitioning of data.
        scaling (bool, optional): If True, apply feature scaling. Defaults to True.
        include_interactions (bool, optional): If True, includes polynomial interaction terms. Defaults to True.
        n_strips (int, optional): Number of spatial strips to divide the data into. Defaults to 3.

    Returns:
        None: The function internally prints out the performance metrics and may save them to files.
    """

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
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),  # , max_depth=20),
        # "Gradient Boosting Machine": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Machine": XGBRegressor(n_estimators=200, random_state=42),
        "Lasso": Lasso(random_state=42),
        "Ridge": Ridge(random_state=42)
    }

    # Results dictionary to store model scores
    results_explained_variance = {name: [] for name in models}
    r2_errors_dict = {name: [] for name in models}
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
            print("Skipped the strip since very few Test or train data in the strip, strip_index=", strip_index)
            sprint(X_train.shape, X_test.shape)
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
            plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
                                     "Spatial_split_train_test_during_model_training" + str(scale) + "_split_" + str(
                                         strip_index) + ".png"), dpi=300)
            plt.show(block=False);
            plt.close()

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
            r2_errors_dict[name].append(r2_scorer(Y_test, cloned_model.predict(X_test)))
            results_mse[name].append(mse)
            if name in models_trained:
                models_trained[name].append(cloned_model)
            else:
                models_trained[name] = [cloned_model]

            output_file_path = os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
                                            f"tod_{tod}_GOF_SPATIAL_CV_SHAPE_of_dataframe_scale_{scale}_.csv")
            with open(output_file_path, "a") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(
                    [cityname, scale, config.CONGESTION_TYPE, tod, "X_train.shape", X_train.shape, "X_test.shape",
                     X_test.shape, "split_index", strip_index])

    for name, score in r2_errors_dict.items():
        print("Inside Spatial CV", config.shift_tile_marker, config.CONGESTION_TYPE, city, scale, name, np.mean(score), "R2")

    # Commented out since SHAP Value computation using the spatial cross validation is not needed.
    # shap_values_list = []
    #
    # output_file_path = os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
    #                                 f"tod_{tod}_GOF_SPATIAL_CV_MEAN_AND_STD_scale_{scale}.csv")
    # with open(output_file_path, "w") as f:
    #     csvwriter = csv.writer(f)
    #     csvwriter.writerow(["Model", "Explained-var-Mean-across-CV", "Explained-var-STD-across-CV",
    #                         "MSE-Mean-across-CV", "MSE-STD-across-CV"] + ["Explained-var-CV-values" + str(x)
    #                                                                       for x in range(1, len(
    #             results_explained_variance[name]) + 1)] +
    #                        ["MSE-var-CV-values" + str(x) for x in range(1, len(results_mse[name]) + 1)])
    #     for name in models:
    #         csvwriter.writerow(
    #             [name, np.mean(results_explained_variance[name]), np.std(results_explained_variance[name]),
    #              np.mean(results_mse[name]), np.std(results_mse[name])] +
    #             results_explained_variance[name] + results_mse[name])
    #
    #     for name in models:
    #         if name == "Random Forest":
    #             print("logging_for_SPATIAL_explained_var", cityname, scale, np.mean(results_explained_variance[name]),
    #                   np.median(results_explained_variance[name]), config.SHAP_mode_spatial_CV, config.CONGESTION_TYPE,
    #                   sep=",")
    #
    # for name in results_explained_variance:
    #     print("Before computing mean: ", name)
    #     print(results_explained_variance[name])
    #     results_explained_variance[name] = np.mean(results_explained_variance[name])
    #     results_mse[name] = np.mean(results_mse[name])
    #
    # print("\n\n-------------------------------------------------------------")
    # print("results_explained_variance spatial")
    # for name, scores in results_explained_variance.items():
    #     print(f"{name}:")
    #     print(f"Scores for each fold: {scores}")
    #     print(f"Average score explained variance: {np.mean(scores)}\n")
    #
    # # Print the results
    # print("\n\n-------------------------------------------------------------")
    # print("results_MSE spatial")
    # for name, scores in results_mse.items():
    #     print(f"{name}:")
    #     print(f"Scores for each fold: {scores}")
    #     print(f"Average score MSE: {np.mean(scores)}\n")
    #
    # if not config.SHAP_values_disabled:
    #     for strip_index in range(n_strips):
    #         a = []
    #         for i in range(len(temp_obj.bbox_X)):
    #             if bbox_to_strip[list(temp_obj.bbox_X[i].keys())[0]] == strip_index:
    #                 a.append(i)
    #
    #         test_mask = X.index.isin(a)
    #         train_mask = ~test_mask
    #
    #         # Split the data into training and test sets based on spatial split
    #         X_train, X_test = X[train_mask], X[test_mask]
    #
    #         try:
    #             if X_train.shape[0] < 5 or X_test.shape[0] < 5:
    #                 print("Skipped the strip since very few Test or train data in the strip, split_counter=",
    #                       strip_index)
    #                 sprint(X_train.shape, X_test.shape)
    #                 continue
    #         except:
    #             # to skip when empty
    #             print("Skipped the strip since very few Test or train data in the strip, split_counter=", strip_index)
    #             continue
    #
    #         # If scaling is required, scale the features
    #         if scaling:
    #             scaler = StandardScaler()
    #             X_train = scaler.fit_transform(X_train)
    #             X_test = scaler.transform(X_test)
    #             X_whole_data_standardised = scaler.transform(X)
    #
    #         rf_model = models_trained["Random Forest"][strip_index]
    #         import shap
    #         explainer = shap.TreeExplainer(rf_model)
    #         # shap_values = explainer.shap_values(X_test)  # # computing shap for test data no post processing needed for statndardisation since the X_test is already standardised
    #         shap_values = explainer.shap_values(
    #             X_whole_data_standardised)  # # computing shap for test data no post processing needed for statndardisation since the X_test is already standardised
    #         shap_values_list.append(shap_values)
    #
    #         if config.FAST_GEN_PDPs_for_multiple_runs:
    #             break  # just a single run for SHAP values for fast prototyping
    #
    #     # Average the SHAP values across all folds
    #     # concatenated_shap_values = np.concatenate(shap_values_list, axis=0)
    #
    #     shap_values_stack = np.stack(shap_values_list, axis=0)
    #
    #     # Compute the mean SHAP values across the first axis (the CV splits axis)
    #     mean_shap_values = np.mean(shap_values_stack, axis=0)
    #
    #     # Aggregate SHAP values
    #     shap_values_agg = mean_shap_values
    #     # Aggregate SHAP values
    #     shap_values_agg = mean_shap_values
    #
    #     # Plot feature importance
    #     total_shap_values = np.abs(shap_values_agg).mean(axis=0)
    #
    #     # Create a list of tuples (feature name, total SHAP value)
    #     feature_importance_list = [(feature, shap_value) for feature, shap_value in zip(X.columns, total_shap_values)]
    #
    #     # Sort the list based on SHAP values in descending order
    #     sorted_feature_importance_list = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)
    #
    #     # Write the sorted list to a file
    #     spatial_folder = os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
    #                                   f"spatial_tod_{tod}_scale_{scale}")
    #     if not os.path.exists(spatial_folder):
    #         os.makedirs(spatial_folder)
    #     output_file_path = os.path.join(spatial_folder, "spatial_total_feature_importance.csv")
    #     with open(output_file_path, "w") as f:
    #         csvwriter = csv.writer(f)
    #         csvwriter.writerow(["Feature", "Total SHAP Value"])
    #         for feature, shap_value in sorted_feature_importance_list:
    #             csvwriter.writerow([feature, shap_value])
    #
    #     plt.clf();
    #     plt.close()
    #     shap.summary_plot(shap_values_agg, X, plot_type="bar", show=False)
    #     plt.title("Feature Importance (Spatial)")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(spatial_folder, "spatial_shap_total_FI.png"))
    #     plt.clf();
    #     plt.close()
    #
    #     # Compute Otsu threshold
    #     otsu_threshold = threshold_otsu(total_shap_values)
    #
    #     # Filter features based on Otsu threshold
    #     # filtered_features = [idx for idx, val in enumerate(total_shap_values) if val > otsu_threshold]
    #     filtered_features = [idx for idx, val in enumerate(total_shap_values) if
    #                          val > 0]  # Removed this filter to plot all cases
    #
    #     for idx in filtered_features:
    #         feature = X.columns[idx]
    #         shap.dependence_plot(feature, shap_values_agg, X, show=False)
    #         plt.ylim(-1, 3)
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(spatial_folder, f"spatial_shap_pdp_{feature}.png"))
    #         plt.clf();
    #         plt.close()
    #
    #     # Write GOF results to a file
    #     gof_file_path = os.path.join(spatial_folder, "GOF_spatial_tod_" + str(tod) + "_scale_" + str(scale) + ".csv")
    #     if not os.path.exists(gof_file_path):
    #         with open(gof_file_path, "w") as f:
    #             csvwriter = csv.writer(f)
    #             csvwriter.writerow(["model", "GoF_explained_Variance", "GoF_MSE", "TOD", "Scale", "cityname"])
    #
    #     with open(gof_file_path, "a") as f:
    #         csvwriter = csv.writer(f)
    #         for name, score in results_explained_variance.items():
    #             csvwriter.writerow([name, score, results_mse[name], tod, scale, cityname])



# Example call to the function
# compare_models_gof_spatial_cv_HPT(X_data, Y_data, features, bbox_to_strip_dict, 'NewYork', 'morning', 100, temp_obj_data)


from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
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
    list_of_cities = "Singapore|Zurich|Mumbai|Auckland|Istanbul|MexicoCity|Bogota|NewYorkCity|Capetown|London".split(
        "|")
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

    common_features = [ 'n',
                     'm',
                     'k_avg',
                     # 'edge_length_total',
                     # 'edge_length_avg',
                     'streets_per_node_avg',
                     # 'street_length_total',
                     # 'street_segment_count',
                     # 'street_length_avg',
                     'circuity_avg',
                     # 'self_loop_proportion',
                     'metered_count',
                     'non_metered_count',
                     'total_crossings',
                     'betweenness',
                     'mean_lanes',
                     # 'lane_density',
                     # 'maxspeed',
                    # 'streets_per_node_count_0',
                    # 'streets_per_node_count_1',
                     # 'streets_per_node_proportion_1',
                     'streets_per_node_count_2',
                     # 'streets_per_node_proportion_2',
                     'streets_per_node_count_3',
                     # 'streets_per_node_proportion_3',
                     'streets_per_node_count_4',
                     # 'streets_per_node_proportion_4',
                     'streets_per_node_count_5',
                     # 'streets_per_node_proportion_5',
                     # 'streets_per_node_count_6',
                     # 'streets_per_node_proportion_6',
                     # 'global_betweenness'
                ]


    all_features = [ 'n',
                     'm',
                     'k_avg',
                     'edge_length_total',
                     'edge_length_avg',
                     'streets_per_node_avg',
                     'street_length_total',
                     'street_segment_count',
                     'street_length_avg',
                     'circuity_avg',
                     'self_loop_proportion',
                     'metered_count',
                     'non_metered_count',
                     'total_crossings',
                     'betweenness',
                     'mean_lanes',
                     'lane_density',
                     'maxspeed',
                     'streets_per_node_count_1',
                     'streets_per_node_proportion_1',
                     'streets_per_node_count_2',
                     'streets_per_node_proportion_2',
                     'streets_per_node_count_3',
                     'streets_per_node_proportion_3',
                     'streets_per_node_count_4',
                     'streets_per_node_proportion_4',
                     'streets_per_node_count_5',
                     'streets_per_node_proportion_5',
                     'streets_per_node_count_6',
                     'streets_per_node_proportion_6',
                     'global_betweenness'
                     ]

    if config.SHAP_use_all_features_including_highly_correlated:
        common_features = all_features

    scale_list = config.scl_list_of_seeds

    for list_of_cities in list_of_cities_list_of_list:
        for tod_list in tod_list_of_list:
            for scale in scale_list:
                for city in list_of_cities:

                    tod = tod_list
                    x = []
                    y = []
                    fname = os.path.join(config.BASE_FOLDER, config.network_folder, city,
                                         f"_scale_{scale}_train_data_{tod}.pkl")
                    try:
                        temp_obj = CustomUnpicklerTrainDataVectors(open(fname, "rb")).load()

                        if config.MASTER_VISUALISE_FEATURE_DIST_INSIDE_ShapAnalysisScript:
                            from urbanscales.preprocessing.prep_network import Scale
                            scl = Scale.get_object(cityname=city, scale=scale)

                        if isinstance(temp_obj.X, pd.DataFrame):
                            #  we only choose the features which exist in this dataframe
                            common_features_list = list(set(temp_obj.X.columns.to_list()).intersection(common_features))



                            filtered_columns = temp_obj.X.filter(
                                regex='streets_per_node_count_|streets_per_node_proportion_').columns
                            # Fill NaN values with 0 in these filtered columns and update the original DataFrame
                            temp_obj.X[filtered_columns] = temp_obj.X[filtered_columns].fillna(0)

                            filtered_X = temp_obj.X[common_features_list]

                            print("#tiles", config.CONGESTION_TYPE, city, scale, temp_obj.X.shape[0])

                            percentage_nan_per_column = temp_obj.X.isna().mean() * 100
                            # Sort these percentages in descending order
                            sorted_percentage_nan_per_column = percentage_nan_per_column.sort_values(ascending=False)
                            # Print the sorted percentages
                            sprint(sorted_percentage_nan_per_column)

                            import seaborn as sns

                            if config.MASTER_VISUALISE_EACH_STEP_INSIDE_ShapAnalysisScript:
                                plt.figure(figsize=(10, 10))
                                corr = (filtered_X).corr()
                                sns.heatmap(corr)
                                if not os.path.exists(config.results_folder):
                                    os.mkdir(config.results_folder)
                                plt.tight_layout()
                                if not os.path.exists(os.path.join(config.BASE_FOLDER, "results")):
                                    os.mkdir(os.path.join(config.BASE_FOLDER, "results"))
                                plt.savefig(os.path.join(config.BASE_FOLDER, "results", "cross-corr-" + slugify(
                                    str((city, str(scale), str(tod))))) + ".png")

                                plt.show(block=False);
                                plt.close()


                    except Exception as e:
                        print("Error in :")
                        print (e)
                        sprint(list_of_cities, tod_list, scale, city)
                        continue


                    def calculate_strip_boundaries_vertical(bboxes, n_strips=7):
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


                    def assign_bboxes_to_strips_vertical(bboxes, strip_boundaries):
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


                    def calculate_strip_boundaries_horizontal(bboxes, n_strips=7):
                        # Extract all the longitude values from the bounding boxes
                        all_lats = [list(bbox.keys())[0][0] for bbox in bboxes] + [list(bbox.keys())[0][1] for bbox
                                                                                   in bboxes]

                        # Sort the longitude values
                        sorted_lats = sorted(list(set(all_lats)))

                        # Calculate the number of bboxes per strip
                        bboxes_per_strip = len(sorted_lats) // n_strips

                        # Initialize the list of strip boundaries
                        strip_boundaries = [sorted_lats[0]]

                        # Determine the boundaries of each strip
                        for i in range(1, n_strips):
                            boundary_index = i * bboxes_per_strip
                            strip_boundaries.append(sorted_lats[boundary_index])

                        # Add the last longitude value as the end boundary of the last strip
                        strip_boundaries.append(sorted_lats[-1])

                        return strip_boundaries


                    def assign_bboxes_to_strips_horizontal(bboxes, strip_boundaries):
                        strip_assignments = {}
                        bbox_to_strip = {}
                        for i, bbox in enumerate(bboxes):
                            bbox_coords = list(bbox.keys())[0]
                            South, North = bbox_coords[1], bbox_coords[
                                0]  # Assuming bbox_coords is in the format (North, South, East, West)

                            # Find the strip that contains the bbox
                            for strip_index in range(len(strip_boundaries) - 1):
                                left_boundary = strip_boundaries[strip_index]
                                right_boundary = strip_boundaries[strip_index + 1]

                                # Check if bbox is within the current strip's boundaries
                                if South >= left_boundary and North <= right_boundary:
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
                                        grid_index = lon_index * n_strips + lat_index  # (lon_index, lat_index)
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
                        if config.MASTER_VISUALISE_EACH_STEP_INSIDE_ShapAnalysisScript:
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
                            # plt.title('Spatial Splits')

                            import geopy.distance

                            # Calculate the average latitude of your bounding boxes
                            a = []
                            latlist = []
                            lonlist = []
                            for i in range(len(bboxes)):
                                a.append(np.mean([list(bboxes[i].keys())[0][0], list(bboxes[i].keys())[0][1]]))
                                latlist.append(list(bboxes[i].keys())[0][0])
                                latlist.append(list(bboxes[i].keys())[0][1])  # The order is NSEW
                                lonlist.append(list(bboxes[i].keys())[0][2])
                                lonlist.append(list(bboxes[i].keys())[0][3])  # The order is NSEW

                            average_latitude = np.mean(a)
                            # sprint (average_latitude)

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
                            plt.title("Spatial Cross validation splits " + city + "_scale_" + str(scale))
                            plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, city,
                                                     "Spatial_splits_for_spatial_CV" + str(scale) + ".png"), dpi=300)
                            plt.show(block=False);
                            plt.close()




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
                        if config.MASTER_VISUALISE_EACH_STEP_INSIDE_ShapAnalysisScript:
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
                            plt.show(block=False);
                            plt.close()


                    bboxes = temp_obj.bbox_X
                    n_strips = 4

                    from shapely.geometry import Polygon

                    if config.MASTER_VISUALISE_EACH_STEP_INSIDE_ShapAnalysisScript:
                        if config.SHAP_mode_spatial_CV == "vertical":
                            n_strips = 6
                            strip_boundaries = calculate_strip_boundaries_vertical(bboxes, n_strips)
                            strip_assignments, bbox_to_strip = assign_bboxes_to_strips_vertical(bboxes, strip_boundaries)
                            visualize_splits(bboxes, strip_boundaries, bbox_to_strip,
                                             split_direction='vertical')
                        elif config.SHAP_mode_spatial_CV == "horizontal":
                            n_strips = 6
                            strip_boundaries = calculate_strip_boundaries_horizontal(bboxes, n_strips)
                            strip_assignments, bbox_to_strip = assign_bboxes_to_strips_horizontal(bboxes, strip_boundaries)
                            visualize_splits(bboxes, strip_boundaries, bbox_to_strip,
                                             split_direction='horizontal')
                        elif config.SHAP_mode_spatial_CV == "grid":
                            lon_boundaries, lat_boundaries = calculate_grid_boundaries(bboxes, n_strips=n_strips)
                            grid_assignments, bbox_to_strip = assign_bboxes_to_grid(bboxes, lon_boundaries, lat_boundaries,
                                                                                    n_strips)
                            visualize_grid(bboxes, lon_boundaries, lat_boundaries, bbox_to_strip, split_direction='grid')
                        else:
                            raise Exception("Wrong split type; must be grid or vertical")

                    # After processing each city and time of day, concatenate data
                    x.append(filtered_X)
                    y.append(temp_obj.Y)

                    # Concatenate the list of DataFrames in x and y
                    assert len(x) == 1
                    X = pd.concat(x, ignore_index=True)
                    # Convert any NumPy arrays in the list to Pandas Series
                    y_series = [pd.Series(array) if isinstance(array, np.ndarray) else array for array in y]

                    # Concatenate the list of Series and DataFrames
                    Y = pd.concat(y_series, ignore_index=True)

                    sprint(city, scale, tod, config.shift_tile_marker, X.shape, Y.shape)

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
                    elif config.SHAP_mode_spatial_CV == "vertical" or config.SHAP_mode_spatial_CV == "horizontal":
                        N_STRIPS = n_strips
                    else:
                        raise Exception("Wrong split type; must be grid or vertical")

                    model_fit_time_start = time.time()

                    assert X.shape[0] == Y.shape[0] # same number of lines in both X and Y

                    # remove the Nan columns
                    X = X.dropna(axis='columns')
                    common_features_list = list(set(common_features_list).intersection(set(X.columns.to_list())))


                    # compare_models_gof_standard_cv_HPT_new(X, Y, common_features_list, tod=tod, cityname=city,
                    #                                        scale=scale,
                    #                                        n_splits=7, include_interactions=False,
                    #                                        scaling=config.SHAP_ScalingOfInputVector)

                    t_non_spatial = time.time() - model_fit_time_start
                    # compare_models_gof_spatial_cv(X, Y, common_features_list, temp_obj=temp_obj, include_interactions=False,
                    #                               bbox_to_strip=bbox_to_strip, n_strips=N_STRIPS, tod=tod, cityname=city,
                    #                               scale=config.SHAP_ScalingOfInputVector)
                    t_spatial = time.time() - model_fit_time_start - t_non_spatial
                    sprint(t_spatial, t_non_spatial, "seconds")

                    if config.MASTER_VISUALISE_FEATURE_DIST_INSIDE_ShapAnalysisScript:
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

                            import matplotlib.pyplot as plt
                            import matplotlib.colors as mcolors

                            # Provided RGB values, arranged from pink to blue
                            colors = [
                                (255, 0, 84),  # Pink
                                (245, 1, 109),
                                (222, 0, 135),
                                (190, 1, 158),
                                (151, 42, 177),
                                (121, 75, 206),
                                (71, 100, 227),
                                (2, 120, 242),
                                (0, 137, 250)  # Blue
                            ]

                            # Normalize RGB values
                            normalized_colors = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in colors][::-1]

                            # Create the colormap
                            custom_cmap_shaplike = mcolors.LinearSegmentedColormap.from_list("CustomGradient", normalized_colors,
                                                                                    N=256)

                            gdf_mercator.plot(ax=ax, column='value', cmap=custom_cmap_shaplike, edgecolor='none',
                                              alpha=0.7)  # Use 'viridis' or any other colormap
                            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                            ax.set_axis_off()
                            plt.title(column)
                            # plt.colorbar()

                            sm = plt.cm.ScalarMappable(cmap=custom_cmap_shaplike,
                                                       norm=plt.Normalize(vmin=np.min(values_normalized),
                                                                          vmax=np.max(values_normalized)))
                            plt.colorbar(sm, ax=ax)

                            plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, city,
                                                     "_feature_" + str(column) + "_" + str(scale) + ".pdf"), dpi=300)
                            plt.show(block=False);
                            plt.close()
                            # break # no need to plot all

                        sys.exit(0)

                       ######################## SAME THING BUT WITH SAME COLOR TO SHOW DIFFERENCES    ########################
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
                                    'geometry': [Polygon([(lon1, lat1), (lon1, lat2), (lon2, lat2), (lon2, lat1)])
                                                 for
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
                            gdf_mercator.plot(ax=ax, column='value', color=(0.121569, 0.466667, 0.705882, 0.1), edgecolor=(0, 0, 0, 1), # using RGBA format for edgecolor
                                            linewidth=0.8)

                            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                            ax.set_axis_off()
                            plt.title(city, fontsize=40)
                            # plt.colorbar()
                            plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, city,
                                                     "_feature_empty_tiles_shifting" + str(config.shift_tile_marker) + str(column) + "_" + str(scale) + ".png"),
                                        dpi=300)
                            plt.show(block=False);
                            plt.close()
                            break # we only need one feature


                        ##########################################################################################
                        ##########################################################################################
                        ##########################################################################################
                        import matplotlib.pyplot as plt
                        import geopandas as gpd
                        from shapely.geometry import Polygon
                        import contextily as ctx
                        import osmnx as ox

                        # Extract the full graph for the total extent

                        # full_graph = ox.simplify_graph(full_graph)

                        # Loop over each feature
                        for column in common_features:
                            # Get the list of bounding boxes
                            bboxes = [list(i.keys())[0] for i in temp_obj.bbox_X]

                            # Get the list of values for the current feature
                            values_list = temp_obj.X[column]

                            # Find the indices of the minimum and maximum values
                            # min_idx = np.argmin(values_list)
                            # max_idx = np.argmax(values_list)
                            sorted_indices = np.argsort(values_list)
                            min_idx = sorted_indices[:5].tolist()  # First 5 indices with min values
                            max_idx = sorted_indices[-5:].tolist()[::-1]

                            for from_top_count in range(3,4):
                                # Create a GeoDataFrame for the minimum and maximum tiles
                                min_tile_gdf = gpd.GeoDataFrame({
                                    'geometry': [Polygon([(bboxes[min_idx[from_top_count]][2], bboxes[min_idx[from_top_count]][0]),
                                                          (bboxes[min_idx[from_top_count]][2], bboxes[min_idx[from_top_count]][1]),
                                                          (bboxes[min_idx[from_top_count]][3], bboxes[min_idx[from_top_count]][1]),
                                                          (bboxes[min_idx[from_top_count]][3], bboxes[min_idx[from_top_count]][0])])],
                                    'value': [values_list[min_idx[from_top_count]]]
                                }, crs="EPSG:4326")
                                from shapely.geometry import Polygon

                                gdf_mercator = min_tile_gdf.to_crs(epsg=3857)
                                fig, ax = plt.subplots(figsize=(10, 10))
                                gdf_mercator.boundary.plot(ax=ax, color='red')
                                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                                ax.set_axis_off()
                                # plt.title(f'{column} - Minimum Subgraph', fontsize=20)
                                plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, city,
                                                         f"min_{column}_subgraph_CTX_" +str(from_top_count)+ ".png"), dpi=300)
                                plt.clf()
                                plt.close()


                                max_tile_gdf = gpd.GeoDataFrame({
                                    'geometry': [Polygon([(bboxes[max_idx[from_top_count]][2], bboxes[max_idx[from_top_count]][0]),
                                                          (bboxes[max_idx[from_top_count]][2], bboxes[max_idx[from_top_count]][1]),
                                                          (bboxes[max_idx[from_top_count]][3], bboxes[max_idx[from_top_count]][1]),
                                                          (bboxes[max_idx[from_top_count]][3], bboxes[max_idx[from_top_count]][0])])],
                                    'value': [values_list[max_idx[from_top_count]]]
                                }, crs="EPSG:4326")
                                from shapely.geometry import Polygon
                                gdf_mercator = max_tile_gdf.to_crs(epsg=3857)
                                fig, ax = plt.subplots(figsize=(10, 10))
                                gdf_mercator.boundary.plot(ax=ax, color='red')
                                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                                ax.set_axis_off()
                                # plt.title(f'{column} - Maximum Subgraph', fontsize=20)
                                plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, city,
                                                         f"max_{column}_subgraph_CTX_" +str(from_top_count)+ ".png"), dpi=300)
                                plt.clf()
                                plt.close()


                                dict_bbox_to_subgraph = scl.dict_bbox_to_subgraph
                                list_of_bbox_min_max = list(((bboxes[max_idx[from_top_count]][2], bboxes[max_idx[from_top_count]][0]),
                                                             (bboxes[max_idx[from_top_count]][2], bboxes[max_idx[from_top_count]][1]),
                                                             (bboxes[max_idx[from_top_count]][3], bboxes[max_idx[from_top_count]][1]),
                                                             (bboxes[max_idx[from_top_count]][3], bboxes[max_idx[from_top_count]][0])))
                                lonlist, latlist = [x[0] for x in list_of_bbox_min_max], [x[1] for x in
                                                                                          list_of_bbox_min_max]
                                N, S, E, W = max(latlist), min(latlist), max(lonlist), min(lonlist)
                                max_subgraph = scl.dict_bbox_to_subgraph[N,S,E,W].G # extracting from the tile

                                dict_bbox_to_subgraph = scl.dict_bbox_to_subgraph
                                list_of_bbox_min_max = list(((bboxes[min_idx[from_top_count]][2], bboxes[min_idx[from_top_count]][0]),
                                                             (bboxes[min_idx[from_top_count]][2], bboxes[min_idx[from_top_count]][1]),
                                                             (bboxes[min_idx[from_top_count]][3], bboxes[min_idx[from_top_count]][1]),
                                                             (bboxes[min_idx[from_top_count]][3], bboxes[min_idx[from_top_count]][0])))
                                lonlist, latlist = [x[0] for x in list_of_bbox_min_max], [x[1] for x in
                                                                                          list_of_bbox_min_max]
                                N, S, E, W = max(latlist), min(latlist), max(lonlist), min(lonlist)
                                min_subgraph = scl.dict_bbox_to_subgraph[N,S,E,W].G # extracting from the tile


                                # Plot the subgraph for the maximum value tile
                                fig, ax = plt.subplots(figsize=(10, 10))

                                # fig, ax = ox.plot_graph(min_subgraph, show=False, close=False, bgcolor="#E7E7E7")
                                ox.plot_graph(max_subgraph, ax=ax, show=False, edge_color='black', edge_linewidth=3)
                                ax.set_axis_off()
                                sprint(max_subgraph.nodes().__len__())
                                # plt.title(f'{column} - Maximum Subgraph', fontsize=20)
                                plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, city,
                                                         f"max_{column}_subgraph_" +str(from_top_count)+ ".pdf"), dpi=300)
                                plt.show()
                                plt.close()

                                fig, ax = plt.subplots(figsize=(10, 10))

                                # fig, ax = ox.plot_graph(min_subgraph, show=False, close=False, bgcolor="#E7E7E7")
                                ox.plot_graph(min_subgraph, ax=ax, show=False, edge_color='black', edge_linewidth=3)
                                ax.set_axis_off()
                                sprint(max_subgraph.nodes().__len__())
                                # plt.title(f'{column} - Minimum Subgraph', fontsize=20)
                                plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, city,
                                                         f"min_{column}_subgraph_" +str(from_top_count)+ ".pdf"), dpi=300)
                                plt.show()
                                plt.close()
                            # break

                        #
                        #

                        ###########################################################################################
                        # ###########################################################################################
                        # import os
                        # import matplotlib.pyplot as plt
                        # import geopandas as gpd
                        # from shapely.geometry import Polygon
                        # import contextily as ctx
                        # import osmnx as ox
                        #
                        # # Set your Stadia Maps API key
                        # api_key = '1844785d-cb2d-4478-bc79-442319986dc8'  # Replace with your actual API key
                        #
                        # # Define the Stadia.StamenTonerBackground tile URL with your API key
                        # # stadia_toner_url = f"https://tiles.stadiamaps.com/tiles/stamen_toner_background/{{z}}/{{x}}/{{y}}.png?api_key={api_key}"
                        # stadia_toner_url = f"https://tiles.stadiamaps.com/tiles/osm_bright/{{z}}/{{x}}/{{y}}.png?api_key={api_key}"
                        #
                        # # Get the full bounding box that covers all the tiles
                        # full_bbox = [
                        #     min([bbox[0] for bbox in bboxes]),  # min latitude
                        #     max([bbox[1] for bbox in bboxes]),  # max latitude
                        #     min([bbox[2] for bbox in bboxes]),  # min longitude
                        #     max([bbox[3] for bbox in bboxes])  # max longitude
                        # ]
                        #
                        # # Extract the full graph for the total extent
                        # full_graph = ox.graph_from_bbox(full_bbox[1], full_bbox[0], full_bbox[3], full_bbox[2],
                        #                                 network_type='drive')
                        #
                        # # Loop over each feature
                        # for column in common_features:
                        #     # Get the list of bounding boxes
                        #     bboxes = [list(i.keys())[0] for i in temp_obj.bbox_X]
                        #
                        #     # Get the list of values for the current feature
                        #     values_list = temp_obj.X[column]
                        #
                        #     # Find the indices of the minimum and maximum values
                        #     min_idx = np.argmin(values_list)
                        #     max_idx = np.argmax(values_list)
                        #
                        #     # Create a GeoDataFrame for the minimum and maximum tiles
                        #     min_tile_gdf = gpd.GeoDataFrame({
                        #         'geometry': [Polygon([(bboxes[min_idx][2], bboxes[min_idx][0]),
                        #                               (bboxes[min_idx][2], bboxes[min_idx][1]),
                        #                               (bboxes[min_idx][3], bboxes[min_idx][1]),
                        #                               (bboxes[min_idx][3], bboxes[min_idx][0])])],
                        #         'value': [values_list[min_idx]]
                        #     }, crs="EPSG:4326")
                        #
                        #     max_tile_gdf = gpd.GeoDataFrame({
                        #         'geometry': [Polygon([(bboxes[max_idx][2], bboxes[max_idx][0]),
                        #                               (bboxes[max_idx][2], bboxes[max_idx][1]),
                        #                               (bboxes[max_idx][3], bboxes[max_idx][1]),
                        #                               (bboxes[max_idx][3], bboxes[max_idx][0])])],
                        #         'value': [values_list[max_idx]]
                        #     }, crs="EPSG:4326")
                        #
                        #     # Convert the GeoDataFrames to the Web Mercator projection
                        #     min_tile_gdf_mercator = min_tile_gdf.to_crs(epsg=3857)
                        #     max_tile_gdf_mercator = max_tile_gdf.to_crs(epsg=3857)
                        #
                        #     # Plotting the minimum value tile with black outline
                        #     fig, ax = plt.subplots(figsize=(10, 10))
                        #     min_tile_gdf_mercator.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8)
                        #     ctx.add_basemap(ax, source=stadia_toner_url)
                        #     ax.set_axis_off()
                        #     plt.title(f'{column} - Minimum Value: {values_list[min_idx]:.2f}', fontsize=20)
                        #     plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, city,
                        #                              f"min_{column}_tile.png"), dpi=300)
                        #     plt.show()
                        #     plt.close()
                        #
                        #     # Extract the subgraph for the minimum value tile from the full graph
                        #
                        #
                        #     # Plotting the maximum value tile with black outline
                        #     fig, ax = plt.subplots(figsize=(10, 10))
                        #     max_tile_gdf_mercator.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8)
                        #     ctx.add_basemap(ax, source=stadia_toner_url)
                        #     ax.set_axis_off()
                        #     plt.title(f'{column} - Maximum Value: {values_list[max_idx]:.2f}', fontsize=20)
                        #     plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, city,
                        #                              f"max_{column}_tile.png"), dpi=300)
                        #     plt.show()
                        #     plt.close()
                        #
                        #     # Extract the subgraph for the maximum value tile from the full graph
                        #
                        #     break
                        # ###########################################################################################
                        # ###########################################################################################
                        #
                        #
                        # import matplotlib.pyplot as plt
                        # import contextily as ctx
                        #
                        #
                        # def add_north_arrow(ax, x=0.05, y=0.95, arrow_length=0.1):
                        #     """Add a north arrow to the plot"""
                        #     ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
                        #                 arrowprops=dict(facecolor='black', width=5, headwidth=15),
                        #                 ha='center', va='center', fontsize=20, xycoords=ax.transAxes)
                        #
                        #
                        # add_north_arrow(ax)
                        # import geopandas as gpd
                        # import matplotlib.pyplot as plt
                        # import contextily as ctx
                        # import numpy as np
                        # from shapely.geometry import Polygon
                        #
                        # # Assuming 'temp_obj', 'common_features', and 'config' are pre-defined as per your setup
                        #
                        # for column in common_features:
                        #     bboxes = [list(i.keys())[0] for i in temp_obj.bbox_X]
                        #     values_list = temp_obj.X[column]
                        #
                        #     # Normalize the values for coloring
                        #     values_normalized = (values_list - np.min(values_list)) / (
                        #                 np.max(values_list) - np.min(values_list))
                        #
                        #     # Create a GeoDataFrame including the values for heatmap
                        #     try:
                        #         gdf = gpd.GeoDataFrame({
                        #             'geometry': [Polygon([(lon1, lat1), (lon1, lat2), (lon2, lat2), (lon2, lat1)]) for
                        #                          lat1, lat2, lon1, lon2 in bboxes],
                        #             'value': values_normalized
                        #         }, crs="EPSG:4326")
                        #     except:
                        #         gdf = gpd.GeoDataFrame({
                        #             'geometry': [Polygon([(lon1, lat1), (lon1, lat2), (lon2, lat2), (lon2, lat1)]) for
                        #                          lat1, lat2, lon1, lon2, _unused_len_ in bboxes],
                        #             'value': values_normalized
                        #         }, crs="EPSG:4326")
                        #
                        #     from geopy.distance import geodesic
                        #     all_sides_distances = []
                        #     min_lat = 99999999999999
                        #     min_lon = 99999999999999
                        #     max_lat = -99999999999999
                        #     max_lon = -99999999999999
                        #     for polygon in gdf['geometry']:
                        #         coords = list(polygon.exterior.coords)
                        #         sides_distances = []
                        #         for i in range(len(coords) - 1):
                        #             point1 = (coords[i][1], coords[i][0])  # (latitude, longitude)
                        #             max_lat = max(max_lat, coords[i][1])
                        #             min_lat = min(min_lat, coords[i][1])
                        #             max_lon = max(max_lon, coords[i][0])
                        #             min_lon = min(min_lon, coords[i][0])
                        #
                        #             point2 = (coords[i + 1][1], coords[i + 1][0])  # (latitude, longitude)
                        #             distance = geodesic(point1, point2).kilometers  # distance in kilometers
                        #             sides_distances.append(distance)
                        #         all_sides_distances.append(sides_distances)
                        #     # Print or store the distances
                        #     total_x_distance = geodesic((min_lat, min_lon), (min_lat, max_lon)).kilometers
                        #     total_y_distance = geodesic((min_lat, min_lon), (max_lat, min_lon)).kilometers
                        #     with open("debug_tile_size.txt", "a") as f2:
                        #         csvwriter = csv.writer(f2)
                        #         csvwriter.writerow([city, scale, np.mean(all_sides_distances), np.std(all_sides_distances), total_x_distance, total_y_distance])
                        #     sprint(city, scale, np.mean(all_sides_distances), np.std(all_sides_distances), total_x_distance, total_y_distance)
                        #
                        #     # Convert the GeoDataFrame to the Web Mercator projection
                        #     gdf_mercator = gdf.to_crs(epsg=3857)
                        #
                        #     # Plotting
                        #     fig, ax = plt.subplots(figsize=(10, 10))
                        #     gdf_mercator.plot(ax=ax, column='value', color=(0.121569, 0.466667, 0.705882, 0.1), edgecolor=(0, 0, 0, 1), # using RGBA format for edgecolor
                        #                     linewidth=0.8)
                        #     ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                        #     ax.set_axis_off()
                        #     plt.title(city)
                        #
                        #     # Adding north arrow
                        #     add_north_arrow(ax)
                        #
                        #     # Save and display
                        #     plt.savefig(os.path.join("./", city +
                        #                              "__feature_empty_tiles_shifting" + str(config.shift_tile_marker) + "_" + str(scale) + "NORTH.png"), dpi=300)
                        #     plt.show(block=False)
                        #     plt.close()
                        #     break
                        #
                        #     # print ("For faster results; Exited after printing one plot per city; comment out if more are needed for other features")
                        #     # sys.exit(0)

                        ###########################################################################################


                        if config.MASTER_VISUALISE_EACH_STEP_INSIDE_ShapAnalysisScript:
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
                            plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, city,
                                                     "_Y_distribution" + str(scale) + ".png"), dpi=300)
                            plt.show(block=False);
                            plt.close()
                        # input("Enter any key to continue for different TOD")
