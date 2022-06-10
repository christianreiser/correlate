from causal_discovery.LPCMCI.compute_experiments import modify_dict_get_graph_and_link_vals
from config import target_label, show_plots, verbosity, random_state, n_measured_links, n_vars_measured, coeff, \
    min_coeff, n_vars_all, n_ini_obs, n_mixed, nth, frac_latents, random_seed, noise_sigma, correct390_0
import math
import numpy as np
import tigramite.data_processing as pp
from matplotlib import pyplot as plt
from tigramite import plotting as tp

# Imports from code inside directory
import generate_data_mod as mod

"""
main challenges to get algo running:
1. 1 -> 1   Modify data generator to start from last sample 10. june
1. 1        intervene in datagenerator                      14. june
2. 5 -> 7   Find optimistic intervention in lpcmci graph    9. june
3. 5        Orient edges with interventional data           22 june
4. 3        Initialize lpcmci with above result             28. june
5. 3        Lpcmci doesn't use data of a variable if it was intervened upon when calculating its causes 4. july

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
    # random_state = np.random.RandomState(random_seed)
    #
    # class NoiseModel:
    #     def __init__(self, sigma=1):
    #         self.sigma = sigma
    #
    #     def gaussian(self, n_samples):
    #         # Get zero-mean unit variance gaussian distribution
    #         return self.sigma * random_state.randn(n_samples)
    #
    # noises = []
    # for link in scm:
    #     noise_type = 'gaussian'
    #     sigma = noise_sigma[0] + (noise_sigma[1] - noise_sigma[0]) * random_state.rand()  # 2,1.2,1,7
    #     noises.append(getattr(NoiseModel(sigma), noise_type))

    ts_check = data_generator(scm, intervention=None, ts_old=[], random_seed=random_seed, n_samples=2000)

    #test
    if ts_check[390,0] == correct390_0:
        print('ts_generated_actual is correct ')
    else:
        print('ts_generated_actual is wrong')

    nonstationary = mod.check_stationarity_chr(ts_check, scm)

    return nonstationary


def generate_stationary_scm():
    """
    generate scms until a stationary one is found
    """
    # todo get settings from config
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
            tau_max=1,
            contemp_fraction=0.6,
            random_state=random_state)  # MT19937(random_state)

        nonstationary = nonstationary_check(scm)  # todo reactivate
        nonstationary = False  # todo remove
        print("nonstationary:", nonstationary, "counter:", counter)
        counter += 1

    # plot scm
    original_graph = plot_scm(scm)  #
    return scm, original_graph


def plot_scm(scm):
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
    if verbosity > 0:
        tp.plot_graph(
            val_matrix=original_vals,  # original_vals None
            link_matrix=original_graph,
            var_names=range(n_vars_all),
            link_colorbar_label='cross-MCI',
            node_colorbar_label='auto-MCI',
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


def data_generator(scm, intervention, ts_old, random_seed, n_samples):
    """
    initialize from last samples of ts
    generate new sample
    intervention=None for observational time series
    output: time series data (might be non-stationary)
    # todo implement interventions
    # todo move configs to config file
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

    ts = mod.generate_nonlinear_contemp_timeseries(links=scm,
                                                   T=n_samples,
                                                   noises=noises,
                                                   random_state=random_state,
                                                   ts_old=ts_old)
    return ts


def measure(ts, obs_vars):
    """
    drop latents
    """
    ts_measured_actual = ts[:, obs_vars]  # remove latents
    ts_measured_actual_df = pp.DataFrame(ts)  # as df
    return ts_measured_actual, ts_measured_actual_df


def obs_discovery(pag_edgemarks, pag_effect_sizes, ts_measured_actual, is_intervention_list):
    """
    1. get observational ts
    2. ini graph with previous pag_edgemarks and pag_effect_sizes
    3. reduce pag_edgemarks with observatonal data and update pag_effect_sizes
    return: pag_edgemarks, pag_effect_sizes
    """
    # todo observational_ts = intersection of ts_measured_actual and is_intervention_list
    ts_observational = ts_measured_actual[not is_intervention_list]
    pag_edgemarks = []  # todo
    pag_effect_sizes = []  # todo
    return pag_edgemarks, pag_effect_sizes


def find_optimistic_intervention(graph_edgemarks, graph_effect_sizes):
    """
    Optimal control to find the most optimistic intervention.
    Optimistic means the PAG is assumed to be correct, and we search for the intervention that maximizes the outcome
    """
    intervention_optimistic = 0  # todo
    return intervention_optimistic


def obs_or_intervene(n_ini_obs, n_mixed, nth):
    """
    first n_ini_obs samples are observational
    then for n_mixed samples, very nth sample is an intervention 
    false: observation
    true: intervention
    """
    is_obs = np.zeros(n_ini_obs).astype(bool)
    is_mixed = np.zeros(n_mixed).astype(bool)
    for i in range(len(is_mixed)):
        if i % nth == 0:
            is_mixed[i] = True
        else:
            is_mixed[i] = False
    is_intervention_list = np.append(is_obs, is_mixed)
    return is_intervention_list


def interv_discovery(ts_measured_actual, pag_edgemarks, pag_effect_sizes, is_intervention_list):
    pag_edgemarks_reduced = []  # todo
    pag_effect_sizes_reduced = []  # todo: not sure how to update
    return pag_edgemarks_reduced, pag_effect_sizes_reduced


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


def main():

    # generate stationary scm
    scm, original_graph = generate_stationary_scm()

    # ini
    ts_generated_actual = np.zeros((0, n_vars_all))
    ts_generated_optimal = []
    ts_measured_actual = []
    is_intervention_list = obs_or_intervene(
        n_ini_obs=n_ini_obs,
        n_mixed=n_mixed,
        nth=nth)  # 500 obs + 500 with every 4th intervention
    n_samples = 1  # len(is_intervention_list)  # todo: extend ts by one sample at a time for len(is_intervention_list)

    measured_labels = np.sort(random_state.choice(range(n_vars_all),  # e.g. [1,4,5,...]
                                                  size=math.ceil(
                                                      (1. - frac_latents) *
                                                      n_vars_all),
                                                  replace=False)).tolist()

    pag_edgemarks = 'fully connected'  # ini PAG as complete graph
    pag_effect_sizes = None
    regret_list = []

    for is_intervention in is_intervention_list[:400]:
        # get interventions of actual PAG and true SCM.
        # output: None if observational or find via optimal control.
        if is_intervention:
            # actual intervention
            intervention_actual = find_optimistic_intervention(pag_edgemarks, pag_effect_sizes)
            # optimal intervention
            true_edgemarks, true_effectsizes = get_edgemarks_and_effect_sizes(scm)
            intervention_optimal = find_optimistic_intervention(true_edgemarks, true_effectsizes),
        else:
            intervention_actual = None
            intervention_optimal = None

        # intervene as proposed and generate new data
        ts_new = data_generator(
            scm=scm,
            intervention=intervention_actual,
            ts_old=ts_generated_actual,
            random_seed=random_seed,
            n_samples=n_samples,
        )

        # append new actual data
        ts_generated_actual = np.r_[ts_generated_actual, ts_new]

    #     # intervene optimally and generate new data
    #     ts_new = data_generator(scm, intervention_optimal, ts_generated_optimal, random_seed, n_samples,
    #                             n_vars_all)
    #     ts_generated_optimal = ts_generated_optimal.append(ts_new)
    #
    #     # measure
    #     ts_measured_actual = ts_measured_actual.append(measure(ts_generated_actual, obs_vars=measured_labels))
    #
    #     # regret
    #     regret_list = np.append(regret_list,
    #                             abs(get_last_outcome(ts_generated_optimal) - get_last_outcome(ts_measured_actual)))
    #
    #     # causal discovery: reduce pag_edgemarks and compute pag_effect_sizes
    #     pag_edgemarks, pag_effect_sizes = obs_discovery(pag_edgemarks='complete_graph', pag_effect_sizes=None,
    #                                                     ts_measured_actual=ts_measured_actual,
    #                                                     is_intervention_list=is_intervention_list)
    #     pag_edgemarks, pag_effect_sizes = interv_discovery(ts_measured_actual, pag_edgemarks, pag_effect_sizes,
    #                                                        is_intervention_list)
    #     pag_edgemarks, pag_effect_sizes = obs_discovery(pag_edgemarks, pag_effect_sizes, ts_measured_actual,
    #                                                     is_intervention_list)
    #
    # regret_sum = sum(regret_list)
    # print('regret_sum:', regret_sum)
    print('done')
    if ts_generated_actual[390,0] == correct390_0:
        print('ts_generated_actual is correct ')
    else:
        print('ts_generated_actual is wrong')
    print('correct390_0:', correct390_0)
    print('done')

main()
