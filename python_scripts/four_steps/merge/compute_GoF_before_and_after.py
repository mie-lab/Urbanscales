# First, a uniform segmentation of the urban space into spatial grids is done and eight graph-based features are extracted for each grid.
# from python_scripts.network_to_elementary.tiles_to_elementary import step_1_osm_tiles_to_features
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from smartprint import smartprint as sprint

import config


def compute_GoF_from_file(filename):
    with open(filename, "rb") as f:
        [X, Y] = pickle.load(f)
        # X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, random_state=int(time.time()))

    # ('scaler', StandardScaler()),
    ("pca", PCA(n_components=8))
    # ("pca", PCA(n_components=12))

    # if "post_merge" in filename:
    Y = Y / np.max(Y)

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA(n_components=config.pca_components)), ("LinR", LinearRegression())]
    )
    # The pipeline can be used as any other estimator
    # and avoids leaking the test set into the train set
    # print(pipe.fit(X_train, y_train))
    # print(pipe.score(X_test, y_test), " GoF measure")

    # Y = Y / 7200 #
    sprint(X.shape, Y.shape)

    X = remove_nans(X)

    pipe.fit(X, Y)
    pipe.score(X, Y)

    # scores = cross_val_score(pipe, X, Y, cv=7)
    # print("CV GoF measure: ", scores)
    # print("Mean CV (GoF):", np.mean(scores))
    # return np.mean(scores)
    # return pipe.score(X_test, y_test)
    # y_pred = pipe.predict(X_test)
    # return mean_squared_error(y_test, y_pred)

    y_pred = pipe.predict(X)
    return mean_squared_error(Y, y_pred)


def remove_nans(X):
    assert (X.shape[1] == 12) and (len(X.shape) == 2)
    column_mean = np.nanmean(X, axis=0)
    indices = np.where(np.isnan(X))
    X[indices] = np.take(column_mean, indices[1])
    return X


if __name__ == "__main__":
    print("Before merge")
    print("MSE: ", compute_GoF_from_file(config.outputfolder + "islands_X_Y"))

    print("After merge")
    print("MSE: ", compute_GoF_from_file(config.outputfolder + "post_merge_X_Y"))
