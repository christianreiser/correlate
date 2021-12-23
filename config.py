"""
config parameters
"""
# paths
private_folder_path = '/home/chrei/code/quantifiedSelfData/'

# target
target_label = 'ProductiveScreentime'  # label of interest

# plots
show_plots = False  # corr matrix
histograms_on = False

# autocorrelation
autocorrelation_on = False

# correlations
load_precomputed_coefficients_and_p_val = False

# features
add_yesterdays_target_feature_on = False
add_ereyesterdays_target_feature_on = True
add_all_yesterdays_features_on = True

# multiple regression
multiple_linear_regression_ensemble_on = False
regularization_strengths = [0.07, 0.07, 0.12]  # 0.07, 0.07, 0.12
sample_weights_on = True
l1_ratios = [1, 0.9, 1]
out_of_bound_correction_on = True
ensemble_weights = [0, 0.4, 0.6]  # [longest, compromise, widest]
phone_vis_height_width = [407, 370]
survey_value_manipulation = False  # to create fake data for visualization survey

# NN
fully_connected_nn_prediction_on = False

# PCA
pca_on = False

# causal discovery
causal_discovery_on = True
tau_max = 32
alpha_level = 0.01
corr_threshold = 0.07
verbosity = 0

# check config
if not sum(ensemble_weights) == 1.0:
    raise ValueError('Config error. Sum(ensemble_weights) != 1.0')

if not add_yesterdays_target_feature_on != add_all_yesterdays_features_on:
    raise ValueError("Config error. Don\'t add add_yesterdays_target_feature twice.")
