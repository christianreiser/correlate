import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tigramite import plotting as tp

import causal_discovery.LPCMCI.generate_data_mod as mod
# Imports from code inside directory
from config import noise_sigma, tau_max, contemp_fraction, verbosity_thesis


def sample_nonzero_cross_dependencies(coeff, min_coeff):
    """
    sample_nonzero_cross_dependencies ~U±(min_coeff and coeff).
    """
    couplings = list(np.arange(min_coeff, coeff + 0.1, 0.1))  # coupling strength
    couplings += [-c for c in couplings]  # add negative coupling strength
    return couplings


def nonstationary_check(scm, random_seed, labels_strs):
    """
    check if scm is stationary
    """
    if verbosity_thesis > 2:
        print('data_generator ...')

    ts_check, health = data_generator(scm, intervention_variable=None,
                              intervention_value=None, ts_old=[], random_seed=random_seed, n_samples=2000,
                              labels=labels_strs,
                              noise_type='gaussian')
    nonstationary = mod.check_stationarity_chr(ts_check, scm)
    return nonstationary


def get_edgemarks_and_effect_sizes(scm):
    n_vars_all = len(scm)

    # ini edgemarks ndarray of size (n_vars, n_vars, tau_max)
    edgemarks = np.full([n_vars_all, n_vars_all, tau_max + 1], '', dtype="U3")

    # ini effect sizes ndarray of size (n_vars, n_vars, tau_max)
    effect_sizes = np.zeros((n_vars_all, n_vars_all, tau_max + 1))

    # iterate over all links in scm
    for affected_var in range(len(scm)):
        # get incoming links on affected var
        affected_var_incoming_links = scm[affected_var]
        # for each incoming links on affected var
        for incoming_link in affected_var_incoming_links:
            # int of causing var
            causal_var = incoming_link[0][0]
            # int of tau with minus
            tau = incoming_link[0][1]
            # effect size
            effect_size = incoming_link[1]

            edgemarks[affected_var, causal_var, -tau] = '<--'
            edgemarks[causal_var, affected_var, -tau] = '-->'
            effect_sizes[affected_var, causal_var, -tau] = effect_size
            effect_sizes[causal_var, affected_var, -tau] = effect_size
    return edgemarks, effect_sizes


def is_cross_dependent_on_target_var(scm):
    """
    check if a different var has an effect on the target var.
    """
    # len = one is the auto dependency
    # > 1 are cross dependencies
    if len(scm[0]) > 1:
        return True
    else:
        return False


def generate_stationary_scm(coeff, min_coeff, random_seed, random_state, n_measured_links, n_vars_measured, n_vars_all,
                            labels_strs):
    """
    generate scms until a stationary one is found
    """
    if verbosity_thesis > 2:
        print('generate_stationary_scm...')
    nonstationary = True
    cross_dependency_on_target_var = False
    scm = []  # stupid ini
    counter = 0
    while nonstationary or not cross_dependency_on_target_var:
        n_links_all = math.ceil(n_measured_links / n_vars_measured * n_vars_all)  # 11

        def lin_f(x):
            return x

        coupling_coeffs = sample_nonzero_cross_dependencies(coeff, min_coeff)
        auto_coeffs = list(np.arange(0.3, 0.6, 0.05))  # somehow error when in config file

        # generate scm
        scm = mod.generate_random_contemp_model(
            N=n_vars_all,  # 11
            L=n_links_all,  # 11
            coupling_coeffs=coupling_coeffs,  # ~U±(min_coeff and coeff) # 0.2,0.3,0.4,0.5,-0.2,-0.3,-0.4,-0.5
            coupling_funcs=[lin_f],
            auto_coeffs=auto_coeffs,  # [0.3, 0.35, 0.4, 0.45, 0.45, 0.55]
            tau_max=tau_max,
            contemp_fraction=contemp_fraction,
            random_state=random_state)  # MT19937(random_state)
        cross_dependency_on_target_var = is_cross_dependent_on_target_var(scm)
        nonstationary = nonstationary_check(scm, random_seed, labels_strs)
        if verbosity_thesis > 1 and counter > 4:
            print("nonstationary / cross_dependency_on_target_var:", nonstationary, '/', cross_dependency_on_target_var,
                  "counter:", counter)
        counter += 1

    # extract true edgemarks, effect sizes from scm
    edgemarks_true, effect_sizes_true = get_edgemarks_and_effect_sizes(scm)

    # plot scm
    plot_scm(edgemarks_true, effect_sizes_true)
    return scm, edgemarks_true, effect_sizes_true


def plot_scm(original_graph, original_vals):
    n_vars_all = len(original_graph)
    if verbosity_thesis > 0:
        # ts_df = pp.DataFrame(ts)

        # save data to file
        # filename = os.path.abspath("./../../../test.dat")
        # fileobj = open(filename, mode='wb')
        # off = np.array(data, dtype=np.float32)
        # off.tofile(fileobj)
        # fileobj.close()

        # plot data
        # if show_plots:
        #     tp.plot_timeseries(ts_df, figsize=(15, 5))
        #     plt.show()

        # plot original DAG
        tp.plot_graph(
            val_matrix=original_vals,  # original_vals None
            link_matrix=original_graph,
            var_names=range(n_vars_all),
            link_colorbar_label='original SCM',
            node_colorbar_label='TODOTODO',
            figsize=(10, 6),
        )
        plt.show()
        # Plot time series graph
        # tp.plot_time_series_graph(
        #     figsize=(12, 8),
        #     val_matrix=original_vals,  # original_vals None
        #     link_matrix=original_graph,
        #     var_names=range(n_vars_all),
        #     link_colorbar_label='MCI',
        # )
        # plt.show()


def measure(ts, obs_vars):
    """
    drop latents
    """
    # drop all columns in ts if their header is not in obs_vars
    ts = ts.drop(list(set(ts.columns) - set(obs_vars)), axis=1)

    # # save ts dataframe to file
    # import os
    # filename = os.path.abspath("./tmp_test.dat")
    # ts.to_csv(filename, index=False)
    return ts


def labels_to_ints(labels, label):
    # get index of label in measured_labels
    # needs tp get the corresponding labels. importnat if latents are included or not
    res = np.where(np.array(labels) == label)[0][0]
    return res


def data_generator(scm,
                   intervention_variable,
                   intervention_value,
                   ts_old,
                   random_seed,
                   n_samples,
                   labels,
                   noise_type):
    """
    initialize from last samples of ts
    generate new sample
    intervention=None for observational time series
    output: time series data (might be non-stationary)
    """

    random_state = np.random.RandomState(random_seed)

    class NoiseModel:
        def __init__(self, sigma=1):
            self.sigma = sigma

        def gaussian(self, n_samples):
            # Get zero-mean unit variance gaussian distribution
            return self.sigma * random_state.randn(n_samples)

    if noise_type == 'gaussian':
        noises = [None]*len(scm)
        for link, link_idx in enumerate(scm):
            sigma = noise_sigma[0] + (noise_sigma[1] - noise_sigma[0]) * random_state.rand()  # 2,1.2,1,7
            noises[link_idx] = getattr(NoiseModel(sigma), noise_type) # check if correct
    elif noise_type == 'without':
        noises = 'without'
    else:
        raise ValueError('noise_type only implemented for \'without\' or "gaussian"')

    # get intervention_var as int. E.g. 'u_0' -> int(0)
    if intervention_variable is not None:
        intervention_variable = labels_to_ints(labels, intervention_variable)

    ts = mod.generate_nonlinear_contemp_timeseries(links=scm,
                                                   T=n_samples,
                                                   noises=noises,
                                                   random_state=random_state,
                                                   ts_old=ts_old,
                                                   intervention_variable=intervention_variable,
                                                   intervention_value=intervention_value)

    # if none, then cyclic contemporaneous scm. then skipp this graph
    if ts is None:
        return None, 'cyclic contemporaneous scm'
    # check if ts is a string

    elif isinstance(ts, str) and ts == 'max_lag == 0':
        return None, ts

    # if ts contains NaNs, value error
    if np.isnan(ts).any():
        raise ValueError("NaN in ts")

    # ts to pandas dataframe and set labels_strs as headers
    ts_df = pd.DataFrame(ts, columns=labels)

    return ts_df, 'good'
