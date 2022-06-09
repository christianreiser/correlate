"""
config parameters
"""
import math

import numpy as np

verbosity = 1

# paths
private_folder_path = '/home/chrei/code/quantifiedSelfData/'

# target
target_label = 'Mood'  # label of interest

# plots
show_plots = False  # corr matrix
histograms_on = False

# autocorrelation
autocorrelation_on = False

# correlations
load_precomputed_coefficients_and_p_val = True

# features
add_yesterdays_target_feature_on = False
add_ereyesterdays_target_feature_on = True
add_all_yesterdays_features_on = True

# multiple regression
multiple_linear_regression_ensemble_on = False
regularization_strengths = [0.07, 0.07, 0.12]  # 0.07, 0.07, 0.12
sample_weights_on = True
l1_ratios = [1, 0.9, 1]
out_of_bound_correction_on = False
ensemble_weights = [0, 0.4, 0.6]  # [longest, compromise, widest]
phone_vis_height_width = [407, 370]
survey_value_manipulation = False  # to create fake data for visualization survey

# NN
fully_connected_nn_prediction_on = False

# PCA
pca_on = False

# causal discovery
causal_discovery_on = True
LPCMCI_or_PCMCI = False  # True for LPCMCI, False for PCMCI
tau_max = 1
# alpha_level = 0.26
# corr_threshold = 0.02
verbosity = 0
pc_alpha = 0.99
remove_link_threshold = 0.08

# check config
if not sum(ensemble_weights) == 1.0:
    raise ValueError('Config error. Sum(ensemble_weights) != 1.0')

if not add_yesterdays_target_feature_on != add_all_yesterdays_features_on:
    raise ValueError("Config error. Don\'t add add_yesterdays_target_feature twice.")

# scm_config
n_vars_measured = 8
random_seed = 0 # todo should not be constant for simulation studies
frac_latents = 0.3
n_measured_links = 8
coeff = 0.5
min_coeff = 0.2
# auto_coeffs = list(np.arange(0.3, 0.6, 0.05)), # somehow error when in config file  # auto-correlations ∼ U(0.3, 0.6) with 0.05 steps  [0.3, 0.35, 0.4, 0.45, 0.45, 0.55]

random_state = np.random.RandomState(random_seed) # MT19937
n_vars_all = math.floor((n_vars_measured / (1. - frac_latents)))#11

# sampling config dict n_ini_obs=500, n_mixed=500, nth=4
n_ini_obs = 500,
n_mixed = 500,
nth = 4