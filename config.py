"""
config parameters
"""
# target
target_label = 'mood'  # label of interest
target_scale_bounds = [1.0, 9.0]

# plots
show_plots = False  # corr matrix
plot_distributions = False

# autocorrelation
autocorrelation_on = False

# correlations
load_precomputed_coefficients_and_p_val = True

# features
add_yesterdays_target_feature = False
add_ereyesterdays_target_feature = True
add_all_yesterdays_features = True

# multiple regression
multiple_linear_regression_ensemble_on = True
regularization_strengths = [0.07, 0.07, 0.12]
l1_ratios = [1, 0.9, 1]
out_of_bound_correction_on = True
ensemble_weights = [0, 0.4, 0.6]  # [longest, compromise, widest]

# NN
fully_connected_nn_prediction_on = False

# PCA
pca_on = True

# check config
if not sum(ensemble_weights) == 1.0:
    raise ValueError('Config error. Sum(ensemble_weights) != 1.0')

if not add_yesterdays_target_feature != add_all_yesterdays_features:
    raise ValueError("Config error. Don\'t add add_yesterdays_target_feature twice")
