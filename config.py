"""
config parameters
"""

target_label = 'mood'  # label of interest
target_scale_bounds = [1.0, 9.0]
show_plots = False  # corr matrix
plot_distributions = False
load_precomputed_coefficients_and_p_val = True
add_yesterdays_target_feature = False
add_ereyesterdays_target_feature = True
add_all_yesterdays_features = True
out_of_bound_correction_on = False
autocorrelation_on = False
multiple_linear_regression_ensemble_on = False
pca_on = False
ensemble_weights = [0, 0.4, 0.6]  # [longest, compromise, widest]
regularization_strength = 0.12
l1_ratio = 1

# check config
if not sum(ensemble_weights) == 1.0:
    raise ValueError('sum(ensemble_weights) != 1.0')
