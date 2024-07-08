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

config.MASTER_VISUALISE_EACH_STEP = True

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


def plot_feature_influence(df, slugifiedstring):
    # Compute MeanSHAP_pos, MeanSHAP_neg, MeanX_pos, MeanX_neg for each feature
    means = {}
    for column in df.columns:
        pos_indices = df[column] > 0
        neg_indices = df[column] < 0

        MeanSHAP_pos = df.loc[pos_indices, column].mean()
        MeanSHAP_neg = df.loc[neg_indices, column].mean()

        # Convert index to a numerical series before calculating mean
        MeanX_pos = pd.Series(df.loc[pos_indices].index).mean()
        MeanX_neg = pd.Series(df.loc[neg_indices].index).mean()

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
        sprint(config.CONGESTION_TYPE, slugifiedstring, i, key, value[2], value[0], value[1])

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

    rf_params = {
        'randomforestregressor__n_estimators': [100, 200, 300, 400, 500, 1000],  # Number of trees
        'randomforestregressor__max_depth': [10, 20, 50, 100],  # Maximum depth of each tree
        # 'randomforestregressor__min_samples_split': [2, 5, 10],
        # Minimum number of samples required to split an internal node
        # 'randomforestregressor__min_samples_leaf': [1, 2, 4, 5],  # Minimum number of samples required at a leaf node
        'randomforestregressor__max_features': ['sqrt', 'log2']
        # Number of features to consider when looking for the best split
    }

    gbm_params = {
        'xgbregressor__n_estimators': [100, 200, 300, 400, 500, 1000],  # Number of boosting stages to perform
        'xgbregressor__learning_rate': [0.001, 0.01, 0.1, 0.2],  # Step size shrinkage used to prevent overfitting
        'xgbregressor__max_depth': [10, 20, 50, 100],  # Maximum depth of each tree
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
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_type="bar")
        shap_values_df = pd.DataFrame(shap_values, columns=feature_list)

        plot_feature_influence(shap_values_df, slugifiedstring=slugify(str((config.CONGESTION_TYPE, cityname, scale, tod))))


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

    shap_values_list = []

    output_file_path = os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
                                    f"tod_{tod}_GOF_SPATIAL_CV_MEAN_AND_STD_scale_{scale}.csv")
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
            if name == "Random Forest":
                print("logging_for_SPATIAL_explained_var", cityname, scale, np.mean(results_explained_variance[name]),
                      np.median(results_explained_variance[name]), config.SHAP_mode_spatial_CV, config.CONGESTION_TYPE,
                      sep=",")

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

    if not config.SHAP_values_disabled:
        for strip_index in range(n_strips):
            a = []
            for i in range(len(temp_obj.bbox_X)):
                if bbox_to_strip[list(temp_obj.bbox_X[i].keys())[0]] == strip_index:
                    a.append(i)

            test_mask = X.index.isin(a)
            train_mask = ~test_mask

            # Split the data into training and test sets based on spatial split
            X_train, X_test = X[train_mask], X[test_mask]

            try:
                if X_train.shape[0] < 5 or X_test.shape[0] < 5:
                    print("Skipped the strip since very few Test or train data in the strip, split_counter=",
                          strip_index)
                    sprint(X_train.shape, X_test.shape)
                    continue
            except:
                # to skip when empty
                print("Skipped the strip since very few Test or train data in the strip, split_counter=", strip_index)
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
            shap_values = explainer.shap_values(
                X_whole_data_standardised)  # # computing shap for test data no post processing needed for statndardisation since the X_test is already standardised
            shap_values_list.append(shap_values)

            if config.FAST_GEN_PDPs_for_multiple_runs:
                break  # just a single run for SHAP values for fast prototyping

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
        spatial_folder = os.path.join(config.BASE_FOLDER, config.network_folder, cityname,
                                      f"spatial_tod_{tod}_scale_{scale}")
        if not os.path.exists(spatial_folder):
            os.makedirs(spatial_folder)
        output_file_path = os.path.join(spatial_folder, "spatial_total_feature_importance.csv")
        with open(output_file_path, "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(["Feature", "Total SHAP Value"])
            for feature, shap_value in sorted_feature_importance_list:
                csvwriter.writerow([feature, shap_value])

        plt.clf();
        plt.close()
        shap.summary_plot(shap_values_agg, X, plot_type="bar", show=False)
        plt.title("Feature Importance (Spatial)")
        plt.tight_layout()
        plt.savefig(os.path.join(spatial_folder, "spatial_shap_total_FI.png"))
        plt.clf();
        plt.close()

        # Compute Otsu threshold
        otsu_threshold = threshold_otsu(total_shap_values)

        # Filter features based on Otsu threshold
        # filtered_features = [idx for idx, val in enumerate(total_shap_values) if val > otsu_threshold]
        filtered_features = [idx for idx, val in enumerate(total_shap_values) if
                             val > 0]  # Removed this filter to plot all cases

        for idx in filtered_features:
            feature = X.columns[idx]
            shap.dependence_plot(feature, shap_values_agg, X, show=False)
            plt.ylim(-1, 3)
            plt.tight_layout()
            plt.savefig(os.path.join(spatial_folder, f"spatial_shap_pdp_{feature}.png"))
            plt.clf();
            plt.close()

        # Write GOF results to a file
        gof_file_path = os.path.join(spatial_folder, "GOF_spatial_tod_" + str(tod) + "_scale_" + str(scale) + ".csv")
        if not os.path.exists(gof_file_path):
            with open(gof_file_path, "w") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(["model", "GoF_explained_Variance", "GoF_MSE", "TOD", "Scale", "cityname"])

        with open(gof_file_path, "a") as f:
            csvwriter = csv.writer(f)
            for name, score in results_explained_variance.items():
                csvwriter.writerow([name, score, results_mse[name], tod, scale, cityname])



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
        [list_of_cities[4]],
        # [list_of_cities[5]],
        # [list_of_cities[6]],
        # [list_of_cities[7]],
        # [list_of_cities[8]],
        # [list_of_cities[9]],
        # list(config.rn_city_wise_bboxes.keys())
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
                     'streets_per_node_count_1',
                     # 'streets_per_node_proportion_1',
                     'streets_per_node_count_2',
                     # 'streets_per_node_proportion_2',
                     'streets_per_node_count_3',
                     # 'streets_per_node_proportion_3',
                     'streets_per_node_count_4',
                     # 'streets_per_node_proportion_4',
                     'streets_per_node_count_5',
                     # 'streets_per_node_proportion_5',
                     'streets_per_node_count_6',
                     # 'streets_per_node_proportion_6',
                     'global_betweenness'  ]


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
                     'global_betweenness'  ]

    if config.SHAP_use_all_features_including_highly_correlated:
        common_features = all_features

    scale_list = [75] # [20, 30, 40, 60, 70, 80, 90, 25, 50, 100]

    for list_of_cities in list_of_cities_list_of_list:
        for tod_list in tod_list_of_list:
            for scale in scale_list:
                for city in list_of_cities:

                    tod = tod_list
                    x = []
                    y = []
                    congestion_type = "probablefixIstanbul"
                    # fname = os.path.join(config.BASE_FOLDER, "/Users/nishant/Downloads/global_betweenness_issue/"+city+"_"+congestion_type,
                    #                      f"_scale_{scale}_train_data_{tod}.pkl")
                    fname = os.path.join(config.BASE_FOLDER,
                                         "/Users/nishant/Downloads/Istanbul_recurrent_rerun_with_correct_global_betweenness/" ,
                                         f"_scale_{scale}_train_data_{tod}.pkl")
                    try:
                        temp_obj = CustomUnpicklerTrainDataVectors(open(fname, "rb")).load()
                        if isinstance(temp_obj.X, pd.DataFrame):
                            #  we only choose the features which exist in this dataframe
                            common_features_list = list(set(temp_obj.X.columns.to_list()).intersection(common_features))



                            filtered_columns = temp_obj.X.filter(
                                regex='streets_per_node_count_|streets_per_node_proportion_').columns
                            # Fill NaN values with 0 in these filtered columns and update the original DataFrame
                            temp_obj.X[filtered_columns] = temp_obj.X[filtered_columns].fillna(0)

                            filtered_X = temp_obj.X[common_features_list]

                            percentage_nan_per_column = temp_obj.X.isna().mean() * 100
                            # Sort these percentages in descending order
                            sorted_percentage_nan_per_column = percentage_nan_per_column.sort_values(ascending=False)
                            # Print the sorted percentages
                            sprint(sorted_percentage_nan_per_column)

                            import seaborn as sns

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
                    # compare_models_gof_spatial_cv(X, Y, common_features, temp_obj=temp_obj, include_interactions=False,
                    #                               bbox_to_strip=bbox_to_strip, n_strips=N_STRIPS, tod=tod, cityname=city,
                    #                               scale=config.SHAP_ScalingOfInputVector)
                    t_spatial = time.time() - model_fit_time_start - t_non_spatial
                    # sprint(t_spatial, t_non_spatial, "seconds")
                    sprint (congestion_type + "_" + city + "_" + str(scale) , X["streets_per_node_count_4"].sum())

                    if False: # Code Folding Trick
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
                            plt.savefig(os.path.join(config.BASE_FOLDER, "results", congestion_type + "_" + city +
                                                     "_feature_" + str(column) + "_" + str(scale) + ".png"), dpi=300)
                            plt.show(block=False);
                            plt.close()

                        if True:
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
                            plt.savefig(os.path.join(config.BASE_FOLDER, "results", congestion_type + "_" + city +
                                                     "_Y_distribution" + str(scale) + ".png"), dpi=300)
                            plt.show(block=False);
                            plt.close()
                        # input("Enter any key to continue for different TOD")

