from causal_discovery.LPCMCI.compute_experiments import modify_dict_get_graph_and_link_vals
from config import target_label, show_plots, verbosity
import math
import numpy as np
import tigramite.data_processing as pp
from matplotlib import pyplot as plt
from tigramite import plotting as tp

# Imports from code inside directory
import generate_data_mod as mod


def sample_nonzero_cross_dependencies(coeff, min_coeff):
    """
    sample_nonzero_cross_dependencies ~U±(min_coeff and coeff).
    """
    couplings = list(np.arange(min_coeff, coeff + 0.1, 0.1))  # coupling strength
    couplings += [-c for c in couplings]  # add negative coupling strength
    return couplings


def generate_scm(random_state, frac_latents, n_measured_links, n_vars_measured, coeff, min_coeff, n_vars_all):
    """
    generate scm given specs
    """
    n_vars_all = math.floor((n_vars_measured / (1. - frac_latents)))  # 11
    n_links_generator = math.ceil(n_measured_links / n_vars_measured * n_vars_all)  # 11

    def lin_f(x):
        return x

    scm = mod.generate_random_contemp_model(
        N=n_vars_all, L=n_links_generator,
        coupling_coeffs=sample_nonzero_cross_dependencies(coeff, min_coeff), # ~U±(min_coeff and coeff)
        coupling_funcs=[lin_f],
        auto_coeffs=list(np.arange(0.3, 0.6, 0.05)),  # auto-correlations ∼ U(0.3, 0.6) with 0.05 steps
        tau_max=1,
        contemp_fraction=0.6,
        random_state=random_state)
    return scm


def data_generator(scm, intervention, ts, random_seed, n_samples, n_vars_all):
    """
    initialize from last samples of ts
    generate new sample
    intervention=None for observational time series
    output: time series data (might be non-stationary)
    # TODO continue from last tau_max samples of ts
    # todo implement interventions
    """
    noise_sigma = (0.5, 2)
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

    ts_check, nonstationary = mod.generate_nonlinear_contemp_timeseries(
        links=scm, T=n_samples + 10000, noises=noises, random_state=random_state)

    if not nonstationary:
        ts = ts_check[:n_samples]
        ts_df = pp.DataFrame(ts)
        original_graph, original_vals = modify_dict_get_graph_and_link_vals(scm)

        # save data to file
        # filename = os.path.abspath("./../../../test.dat")
        # fileobj = open(filename, mode='wb')
        # off = np.array(data, dtype=np.float32)
        # off.tofile(fileobj)
        # fileobj.close()

        # plot data
        if show_plots:
            tp.plot_timeseries(ts_df, figsize=(15, 5))
            plt.show()

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
        return ts, ts_df, original_graph
    else:
        print('nonstationary')


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
    # ini
    random_seed = 0
    random_state = np.random.RandomState(random_seed)

    n_vars_measured = 8
    frac_latents = 0.3
    n_vars_all = math.floor((n_vars_measured / (1. - frac_latents)))
    scm = generate_scm(random_state=random_state, frac_latents=frac_latents, n_measured_links=8, n_vars_measured=8,
                       coeff=0.5, min_coeff=0.2, n_vars_all=n_vars_all)
    ts_generated_actual = []
    ts_generated_optimal = []
    ts_measured_actual = []
    is_intervention_list = obs_or_intervene(n_ini_obs=500, n_mixed=500, nth=4) # 500 obs + 500 with every 4th intvention
    n_samples = len(is_intervention_list)  # todo: extend ts by one sample at a time for len(is_intervention_list)

    measured_labels = np.sort(random_state.choice(range(n_vars_all),  # e.g. [1,4,5,...]
                                                  size=math.ceil((1. - frac_latents) * n_vars_all),
                                                  replace=False)).tolist()

    pag_edgemarks = 'fully connected'  # ini PAG as complete graph
    pag_effect_sizes = None
    regret_list = []

    for is_intervention in is_intervention_list:
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

        # intervene and generate new data
        ts_generated_actual = ts_generated_actual.append(
            data_generator(scm=scm, intervention=intervention_actual, ts=ts_generated_actual, random_seed=random_seed,
                           n_samples=n_samples,
                           n_vars_all=n_vars_all))
        ts_generated_optimal = ts_generated_optimal.append(
            data_generator(scm, intervention_optimal, ts_generated_optimal, random_seed, n_samples, n_vars_all))

        # measure
        ts_measured_actual = ts_measured_actual.append(measure(ts_generated_actual, obs_vars=measured_labels))

        # regret
        regret_list = np.append(regret_list,
                                abs(get_last_outcome(ts_generated_optimal) - get_last_outcome(ts_measured_actual)))

        # causal discovery: reduce pag_edgemarks and compute pag_effect_sizes
        pag_edgemarks, pag_effect_sizes = obs_discovery(pag_edgemarks='complete_graph', pag_effect_sizes=None,
                                                        ts_measured_actual=ts_measured_actual,
                                                        is_intervention_list=is_intervention_list)
        pag_edgemarks, pag_effect_sizes = interv_discovery(ts_measured_actual, pag_edgemarks, pag_effect_sizes,
                                                           is_intervention_list)
        pag_edgemarks, pag_effect_sizes = obs_discovery(pag_edgemarks, pag_effect_sizes, ts_measured_actual,
                                                        is_intervention_list)

    regret_sum = sum(regret_list)
    print('regret_sum:', regret_sum)

main()