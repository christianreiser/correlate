"""
config parameters
"""
# paths
private_folder_path = '/home/chrei/code/quantifiedSelfData/'

# target
target_label = 'Mood'  # label of interest

# plots
show_plots = False  # corr matrix
plot_distributions = False

# autocorrelation
autocorrelation_on = True

# correlations
load_precomputed_coefficients_and_p_val = True

# features
add_yesterdays_target_feature = False
add_ereyesterdays_target_feature = True
add_all_yesterdays_features = True

# multiple regression
multiple_linear_regression_ensemble_on = True
regularization_strengths = [0.07, 0.07, 0.12]
sample_weights_on = True
l1_ratios = [1, 0.9, 1]
out_of_bound_correction_on = True
ensemble_weights = [0, 0.4, 0.6]  # [longest, compromise, widest]
phone_vis_height_width = [407, 370]
survey_value_manipulation = False  # todo remove after survey

# NN
fully_connected_nn_prediction_on = False

# PCA
pca_on = False

# check config
if not sum(ensemble_weights) == 1.0:
    raise ValueError('Config error. Sum(ensemble_weights) != 1.0')

if not add_yesterdays_target_feature != add_all_yesterdays_features:
    raise ValueError("Config error. Don\'t add add_yesterdays_target_feature twice")
