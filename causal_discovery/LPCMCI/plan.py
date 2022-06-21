print('import...')
import math
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tigramite import plotting as tp

# Imports from code inside directory
import generate_data_mod as mod
from causal_discovery.LPCMCI.compute_experiments import modify_dict_get_graph_and_link_vals
from causal_discovery.LPCMCI.observational_discovery import observational_causal_discovery
from causal_discovery.interventional_discovery import get_independencies_from_interv_data
from config import target_label, verbosity_thesis, random_state, n_measured_links, n_vars_measured, coeff, \
    min_coeff, n_vars_all, n_ini_obs, n_mixed, nth, frac_latents, random_seed, noise_sigma, tau_max, \
    contemp_fraction, labels_strs
from intervention_proposal.propose_from_eq import drop_unintervenable_variables, find_most_optimistic_intervention
from intervention_proposal.target_eqs_from_pag import plot_graph, compute_target_equations, \
    make_redundant_information_with_symmetry

"""
next todo:
background: interventional discovery should be analogous to lpcmci, but lpcmci uses interventional as input. 
-> first how lpcmci works and check if it's in and output formats are the same. 
because then i could use the same output for interventional discovery and use it as input for lpcmci

- understand how lpcmci works between link_list, get links, CI test, edge removal/orientation because it's needed for interventional discovery
- build interventional discovery analogous to lpcmci steps above
- should the interface between lpcmci and interventional discovery be links_list or graph (e.g. 'oL>')?
- do i have to add into the dict to say that sth belongs to G ('-' edgemark) or that result is not adjacent ("")


"""
"""
main challenges to get algo running:
1. 1 -> 1   Modify data generator to start from last sample 10. june
1. 1 -> 2   intervene in datagenerator                      14->15. june
2. 5 -> 7   Find optimistic intervention in lpcmci graph    9. june
5. 3 -> 1   Lpcmci doesn't use data of a variable if it was intervened upon when calculating its causes 
3. 5        Orient edges with interventional data           22 june -> 21. june
                ini complete graph
                    mby similar as 'link_list = ' 
                for each intervened var do CI tests and remove edges
4. 3        Initialize lpcmci with above result at all inis 28. june -> 21. june
                (mby replace 'link_list = ' occurrences)            

further TODOs
1. 2        compute optimal intervention from SCM (for ground truth)6. july
2. 2        calculate regret 11. july
3. 5        set up simulation study 19. july
4. 5        interpret results 27. july
5. 40       write 5. oct

-> 27 coding days + 40 writing days = 57 days = 11.5 weeks = 3 months (optimistic guess) 
    => 3.75 with phinc => end of september

-> 75 coding days + 60 writing days = 135 days = 22 weeks = 5.5 months (guess delay factor: 2.8x coding, 1.5x writing) 
    => 7 with phinc => end of end of year
"""


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
    ts_check = data_generator(scm, intervention_variable=None,
                              intervention_value=None, ts_old=[], random_seed=random_seed, n_samples=2000)
    nonstationary = mod.check_stationarity_chr(ts_check, scm)
    return nonstationary


def generate_stationary_scm():
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
        original_graph, original_vals = modify_dict_get_graph_and_link_vals(scm)

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


def data_generator(scm,
                   intervention_variable,
                   intervention_value,
                   ts_old,
                   random_seed,
                   n_samples):
    """
    initialize from last samples of ts
    generate new sample
    intervention=None for observational time series
    output: time series data (might be non-stationary)
    # todo implement interventions
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
        intervention_variable = int(intervention_variable[2:])

    ts = mod.generate_nonlinear_contemp_timeseries(links=scm,
                                                   T=n_samples,
                                                   noises=noises,
                                                   random_state=random_state,
                                                   ts_old=ts_old,
                                                   intervention_variable=intervention_variable,
                                                   intervention_value=intervention_value)

    # ts to pandas dataframe and set labels_strs as headers
    ts_df = pd.DataFrame(ts, columns=labels_strs)

    return ts_df


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


def get_intervention_value(var_name, intervention_coeff, ts_measured_actual):
    ts_measured_actual = pd.DataFrame(ts_measured_actual)
    intervention_value = 0  # ini
    intervention_idx = var_name[2:]  # 'u_0' -> '0'
    intervention_var_measured_values = ts_measured_actual[intervention_idx]
    # get 90th percentile of intervention_var_measured_values
    if intervention_coeff > 0:
        intervention_value = np.percentile(intervention_var_measured_values, 90)
    elif intervention_coeff < 0:
        intervention_value = np.percentile(intervention_var_measured_values, 10)
    else:
        ValueError("intervention_coeff must be positive or negative")
    return intervention_value


def load_eq():
    # load target_ans_per_graph_dict and graph_combinations from file via pickle
    with open('/home/chrei/PycharmProjects/correlate/intervention_proposal/target_eq_simulated.pkl', 'rb') as f:
        target_eq = pickle.load(f)
    with open('/home/chrei/PycharmProjects/correlate/intervention_proposal/graph_combinations_simulated.pkl',
              'rb') as f:
        graph_combinations = pickle.load(f)
    print("attention: target_eq and graph_combinations loaded from file")
    return target_eq, graph_combinations


def find_optimistic_intervention(graph_edgemarks, graph_effect_sizes, measured_labels, ts_measured_actual):
    """
    Optimal control to find the most optimistic intervention.
    """
    # get target equations from graph
    target_eq, graph_combinations = compute_target_equations(
        val_min=graph_effect_sizes,
        graph=graph_edgemarks,
        var_names=measured_labels)

    # load eq instead of calculating them
    # target_eq, graph_combinations = load_eq()

    # remove unintervenable variables
    target_eqs_intervenable = drop_unintervenable_variables(target_eq)

    # get optimal intervention
    largest_abs_coeff, best_intervention_var_name, most_optimistic_graph_idx, intervention_coeff = find_most_optimistic_intervention(
        target_eqs_intervenable)


    # tp.plot_graph(
    #     val_matrix=graph_effect_sizes,
    #     link_matrix=make_redundant_information_with_symmetry(graph_combinations[most_optimistic_graph_idx]),
    #     var_names=measured_labels,
    #     link_colorbar_label='current estimate with ambiguities',
    #     # node_colorbar_label='auto-MCI',
    #     figsize=(10, 6),
    # )
    # plt.show()

    # most optimistic graph
    most_optimistic_graph = graph_combinations[most_optimistic_graph_idx]

    # plot most optimistic graph
    if verbosity_thesis > 0:
        plot_graph(graph_effect_sizes, most_optimistic_graph, measured_labels, 'most optimistic')

    intervention_value = get_intervention_value(best_intervention_var_name, intervention_coeff, ts_measured_actual)
    return best_intervention_var_name, intervention_value


def obs_or_intervene(
        # n_ini_obs,
        n_mixed,
        nth):
    """
    first n_ini_obs samples are observational
    then for n_mixed samples, very nth sample is an intervention
    false: observation
    true: intervention
    """
    # is_obs = np.zeros(n_ini_obs).astype(bool)
    is_mixed = np.zeros(n_mixed).astype(bool)
    for i in range(len(is_mixed)):
        if i % nth == 0:
            is_mixed[i] = True
        else:
            is_mixed[i] = False
    # is_intervention_list = np.append(is_obs, is_mixed)
    return is_mixed


def get_last_outcome(ts_measured_actual):
    """
    in the last sample of ts_measured_actual get value of the target_label
    """
    outcome_last = ts_measured_actual[-1][target_label]  # todo check if it works
    return outcome_last


def get_edgemarks_and_effect_sizes(scm):
    edgemarks = scm['edgemarks']  # todo
    effect_sizes = scm['effect_sizes']  # todo
    return edgemarks, effect_sizes


def get_measured_labels():
    measured_labels = np.sort(random_state.choice(range(n_vars_all),  # e.g. [1,4,5,...]
                                                  size=math.ceil(
                                                      (1. - frac_latents) *
                                                      n_vars_all),
                                                  replace=False)).tolist()
    # measured_labels to strings
    measured_labels = [str(x) for x in measured_labels]
    return measured_labels


def store_intervention(was_intervened, intervention_variable):
    """
    add data to boolean array of measured variables indicating if they were intervened upon
    input: requires that intervention_variable is a string of the form 'char char int' e.g. 'u_0'
    """

    new_series = pd.Series(np.zeros(n_vars_measured, dtype=bool), index=was_intervened.columns)

    # if intervened
    if intervention_variable is not None:
        # get ind
        intervention_idx = intervention_variable[2:]
        # mark intervened var
        new_series[intervention_idx] = True

    # append new_series to was_intervened
    was_intervened = was_intervened.append(new_series, ignore_index=True)

    # # save was_intervened dataframe to file
    # import os
    # filename = os.path.abspath("./tmp_was_intervened.dat")
    # was_intervened.to_csv(filename, index=False)
    return was_intervened


def main():
    # generate stationary scm
    scm, original_graph = generate_stationary_scm()

    # ini
    ts_generated_actual = np.zeros((0, n_vars_all))
    ts_generated_actual = np.zeros((0, n_vars_all))
    is_intervention_list = obs_or_intervene(
        # n_ini_obs=n_ini_obs,
        n_mixed=n_mixed,
        nth=nth)  # 500 obs + 500 with every 4th intervention
    n_samples = 1  # len(is_intervention_list)

    measured_labels = get_measured_labels()

    """ observe first 500 samples"""
    # generate observational data
    ts_df = data_generator(
        scm=scm,
        intervention_variable=None,
        intervention_value=None,
        ts_old=ts_generated_actual,
        random_seed=random_seed,
        n_samples=n_ini_obs[0],
    )

    # measure new data
    ts_measured_actual = measure(ts_df, obs_vars=measured_labels)

    # keep track of where the intervention is
    was_intervened = pd.DataFrame(np.zeros((n_ini_obs[0], n_vars_measured), dtype=bool), columns=measured_labels)

    pag_effect_sizes, pag_edgemarks = observational_causal_discovery(external_independencies=None,
                                                                     df=ts_measured_actual,
                                                                     was_intervened=was_intervened)
    # pag_effect_sizes, pag_edgemarks, var_names = load_results(name_extension='simulated')

    """ loop: causal discovery, planning, intervention """
    for is_intervention in is_intervention_list:
        # get interventions of actual PAG and true SCM.
        # output: None if observational or find via optimal control.
        if is_intervention:
            # actual intervention
            intervention_variable, intervention_value = find_optimistic_intervention(pag_edgemarks, pag_effect_sizes,
                                                                                     measured_labels,
                                                                                     ts_measured_actual)
            # keep track of where the intervention is
            was_intervened = store_intervention(was_intervened, intervention_variable)

            # optimal intervention todo
            # true_edgemarks, true_effectsizes = get_edgemarks_and_effect_sizes(scm)
            # intervention_optimal = find_optimistic_intervention(true_edgemarks, true_effectsizes),
        else:
            intervention_variable = None
            intervention_value = None
            was_intervened = store_intervention(was_intervened, intervention_variable)

        # intervene as proposed and generate new data
        ts_new = data_generator(
            scm=scm,
            intervention_variable=intervention_variable,
            intervention_value=intervention_value,
            ts_old=ts_generated_actual,
            random_seed=random_seed,
            n_samples=n_samples,
        )

        # append new actual data
        ts_generated_actual = np.r_[ts_generated_actual, ts_new]

        # measure new data
        new_measurements = measure(ts_new, obs_vars=measured_labels)

        # append new measured data
        ts_measured_actual = pd.DataFrame(np.r_[ts_measured_actual, new_measurements], columns=measured_labels)

        #     # intervene optimally and generate new data
        #     ts_new = data_generator(scm, intervention_optimal, ts_generated_optimal, random_seed, n_samples,
        #                             n_vars_all)
        #     ts_generated_optimal = ts_generated_optimal.append(ts_new)
        #

        #
        #     # regret
        #     regret_list = np.append(regret_list,
        #                             abs(get_last_outcome(ts_generated_optimal) - get_last_outcome(ts_measured_actual)))
        #
        # causal discovery: reduce pag_edgemarks and compute pag_effect_sizes
        #

        independencies_from_interv_data = get_independencies_from_interv_data(ts_measured_actual, was_intervened)

        pag_effect_sizes, pag_edgemarks = observational_causal_discovery(df=ts_measured_actual,
                                                                         was_intervened=was_intervened,
                                                                         external_independencies=independencies_from_interv_data)
    #
    # regret_sum = sum(regret_list)
    # print('regret_sum:', regret_sum)

    print('done')


main()
