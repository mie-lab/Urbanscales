# Four Steps in abstract
"""
. First, a uniform segmentation of the urban space into spatial grids is done and eight graph-based features are extracted for each grid.
. Second, we use these features as independent variables in a regression model and obtain a goodness-of-fit (GOF) when congestion duration is used as the dependent variable. Feature selection is incorporated into the second step to retain the top N features.
. Third, we vary the size of the grids and evaluate the GOF for different grid sizes.
. Finally, in the fourth step, we manipulate the grids using an iterative split-merge algorithm to reallocate the segmentation of urban space into unconstrained regions, such that the GOF is maximised.
"""


gn1d = 30
