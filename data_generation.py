import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tigramite import plotting as tp

import causal_discovery.LPCMCI.generate_data_mod as mod
# Imports from code inside directory
from causal_discovery.LPCMCI.compute_experiments import scm_to_graph
from config import noise_sigma, labels_strs, n_vars_all, random_seed, n_measured_links, n_vars_measured, \
    tau_max, contemp_fraction, random_state, verbosity_thesis


def sample_nonzero_cross_dependencies(coeff, min_coeff):
    """
    sample_nonzero_cross_dependencies ~U±(min_coeff and coeff).
    """
    couplings = list(np.arange(min_coeff, coeff + 0.1, 0.1))  # coupling strength
    couplings += [-c for c in couplings]  # add negative coupling strength
    return couplings


def nonstationary_check(scm):
    """
    check if scm is stationary
    """
    print('data_generator ...')

    ts_check = data_generator(scm, intervention_variable=None,
                              intervention_value=None, ts_old=[], random_seed=random_seed, n_samples=2000, labels_strs=labels_strs)
    nonstationary = mod.check_stationarity_chr(ts_check, scm)
    return nonstationary


def generate_stationary_scm(coeff, min_coeff):
    """
    generate scms until a stationary one is found
    """
    print('generate_stationary_scm...')
    nonstationary = True
    scm = []  # stupid ini
    counter = 0
    while nonstationary:
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

        nonstationary = nonstationary_check(scm)
        print("nonstationary:", nonstationary, "counter:", counter)
        counter += 1

    # plot scm
    original_graph = plot_scm(scm)  #
    return scm, original_graph


def plot_scm(scm):
    if verbosity_thesis > 0:
        # ts_df = pp.DataFrame(ts)
        original_graph, original_vals = scm_to_graph(scm)

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
            node_colorbar_label='auto-',
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

        return original_graph


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


def data_generator(scm,
                   intervention_variable,
                   intervention_value,
                   ts_old,
                   random_seed,
                   n_samples,
                   labels):
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

    noises = []
    for link in scm:
        noise_type = 'gaussian'
        sigma = noise_sigma[0] + (noise_sigma[1] - noise_sigma[0]) * random_state.rand()  # 2,1.2,1,7
        noises.append(getattr(NoiseModel(sigma), noise_type))

    # get intervention_var as int. E.g. 'u_0' -> int(0)
    if intervention_variable is not None:
        # if len >2 then there is the u_ prefix
        if len(intervention_variable)>2:
            intervention_variable = int(intervention_variable[2:])
        else:
            intervention_variable = int(intervention_variable)

    ts = mod.generate_nonlinear_contemp_timeseries(links=scm,
                                                   T=n_samples,
                                                   noises=noises,
                                                   random_state=random_state,
                                                   ts_old=ts_old,
                                                   intervention_variable=intervention_variable,
                                                   intervention_value=intervention_value)

    # ts to pandas dataframe and set labels_strs as headers
    ts_df = pd.DataFrame(ts, columns=labels)

    return ts_df
