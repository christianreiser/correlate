import math
import os
import pickle
import random as rd
import time

import numpy as np
import tigramite.data_processing as pp
from matplotlib import pyplot as plt
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr

# Imports from code inside directory
import generate_data_mod as mod
import utilities as utilities
from causal_discovery.LPCMCI import metrics_mod
from lpcmci import LPCMCI

# Directory to save results
folder_name = "results/"

samples = 1  # int number of time series realizations to generate
verbosity = 0  # verbosity
config_list = ['random_lineargaussian-8-8-0.2-0.5-0.5-0.6-0.3-1-500-par_corr-lpcmci_nprelim4-0.26-1']  # string that identifies a particular experiment consisting of a model and method.
num_configs = len(config_list)

time_start = time.time()

if verbosity > 0:
    plot_data = True
else:
    plot_data = False


def modify_dict_get_graph_and_link_vals(original_dict):
    """
    outputs:
    1. new dict with link
    2. link graph
    3. val graph

    input:
    dict with format s.th. like
    my_dict = {0: [((0, -1), 0.85, 'remove'),
                   ((1, 0), -0.5, 'remove'),
                   ((2, -1), 0.7, 'remove')],
               1: [((1, -1), 0.8, 'remove'),
                   ((2, 0), 0.7, 'remove')],
               2: [((2, -1), 0.9, 'remove')],
               3: [((3, -2), 0.8, 'remove'),
                   ((0, -3), 0.4, 'remove')]}
    """
    my_dict = original_dict.copy() # otherwise it changes scm_dict in main
    len_dict = len(my_dict)
    max_time_lag = 0

    for key in my_dict:
        my_list = my_dict[key]
        len_my_list = len(my_list)
        modified_list = []
        for list_index in range(len_my_list):
            my_tuple = my_list[list_index]
            modified_tuple = my_tuple[:-1]
            modified_list.append(modified_tuple)

            # get max time lag
            if max_time_lag > modified_tuple[0][1]:
                max_time_lag = modified_tuple[0][1]

        my_dict.update({key: modified_list})

    # print('links:', my_dict)

    max_time_lag = - max_time_lag

    graph = np.ndarray(shape=(len_dict, len_dict, max_time_lag + 1), dtype='U3')
    val = np.zeros(shape=(len_dict, len_dict, max_time_lag + 1), dtype=float)
    for key in my_dict:
        my_list = my_dict[key]
        len_my_list = len(my_list)
        for list_index in range(len_my_list):
            my_tuple = my_list[list_index]
            effected_index = key
            cause_index = my_tuple[0][0]
            lag = -my_tuple[0][1]
            link_strength = my_tuple[1]
            graph[cause_index][effected_index][lag] = '-->'
            if lag == 0:
                graph[effected_index][cause_index][lag] = '<--'
            val[effected_index][cause_index][lag] = link_strength
            val[cause_index][effected_index][lag] = link_strength
    return graph, val



def generate_data(random_state, links, noise_types, noise_sigma, model, T):
    """
    input: links of SCM
    output: time series data (might be non-stationary)
    """
    class NoiseModel:
        def __init__(self, sigma=1):
            self.sigma = sigma

        def gaussian(self, T):
            # Get zero-mean unit variance gaussian distribution
            return self.sigma * random_state.randn(T)

    noises = []
    for j in links:
        noise_type = random_state.choice(noise_types)  # gaussian
        sigma = noise_sigma[0] + (noise_sigma[1] - noise_sigma[0]) * random_state.rand()  # 2,1.2,1,7
        noises.append(getattr(NoiseModel(sigma), noise_type))

    data_all_check = mod.generate_nonlinear_contemp_timeseries(links=links, T=T + 10000, noises=noises,
                                                                              random_state=random_state)
    nonstationary = mod.check_stationarity_chr(data_all_check, links)
    return nonstationary, data_all_check


def generate_fixed_data():
    seed = 7
    auto_coeff = 0.95
    coeff = 0.4
    T = 500

    def lin(x): return x

    links = {0: [((0, -1), auto_coeff, lin),
                 ((1, -1), -coeff, lin)
                 ],
             1: [((1, -1), auto_coeff, lin),
                 ],
             }

    # Specify dynamical noise term distributions, here unit variance Gaussians
    random_state = np.random.RandomState(seed)
    noises = noises = [random_state.randn for j in links.keys()]

    data, nonstationarity_indicator = pp.structural_causal_process(
        links=links, T=T, noises=noises, seed=seed)
    T, N = data.shape

    # Initialize dataframe object, specify variable names
    var_names = [j for j in range(N)]
    dataframe = pp.DataFrame(data, var_names=var_names)

    filename = os.path.abspath("test.dat")
    fileobj = open(filename, mode='wb')
    off = np.array(data, dtype=np.float32)
    off.tofile(fileobj)
    fileobj.close()

    return dataframe, links, var_names


def generate_dataframe(model, coeff, min_coeff, auto, sam, N, frac_unobserved, n_links, max_true_lag, T,
                       contemp_fraction):
    """
    Generate dataframe and links of SCM
    1. generates links from input data about model (mod.generate_random_contemp_model(...))
    2. generates df from links (generate_data)
    3. drops non-stationary data
    :param model:
    :param coeff:
    :param min_coeff:
    :param auto:
    :param sam:
    :param N:
    :param frac_unobserved:
    :param n_links:
    :param max_true_lag:
    :param T:
    :param contemp_fraction:
    :return: dataframe, links, observed_vars, original_graph

    """
    def lin_f(x):
        return x

    def f2(x):
        return x + 5. * x ** 2 * np.exp(-x ** 2 / 20.)

    # noise
    coupling_funcs = [lin_f]
    noise_types = ['gaussian']  # , 'weibull', 'uniform']
    noise_sigma = (0.5, 2)


    couplings = list(np.arange(min_coeff, coeff + 0.1, 0.1))  # coupling strength
    couplings += [-c for c in couplings]  # add negative coupling strength

    auto_deps = list(np.arange(0.3, 0.6, 0.05))  # auto-correlations

    # Models may be non-stationary. Hence, we iterate over a number of seeds
    # to find a stationary one regarding network topology, noises, etc
    if verbosity > 999:
        model_seed = verbosity - 1000
    else:
        model_seed = sam

    for ir in range(1000):
        random_state = np.random.RandomState(0)# todo (model_seed)

        N_all = math.floor((N / (1. - frac_unobserved)))  # 4
        n_links_all = math.ceil(n_links / N * N_all)  # 4
        observed_vars = np.sort(random_state.choice(range(N_all),  # [1,2,3]
                                                    size=math.ceil((1. - frac_unobserved) * N_all),
                                                    replace=False)).tolist()

        links = mod.generate_random_contemp_model(
            N=N_all,
            L=n_links_all,
            coupling_coeffs=couplings,
            coupling_funcs=coupling_funcs,
            auto_coeffs=auto_deps,
            tau_max=max_true_lag,
            contemp_fraction=contemp_fraction,
            # num_trials=1000,
            random_state=random_state)

        # generate data from links
        nonstationary, data_all_check = generate_data(random_state, links, noise_types, noise_sigma, model, T)

        # If the model is stationary, break the loop
        if not nonstationary:
            data_all = data_all_check[:T]
            dataframe_all = pp.DataFrame(data_all)
            data = data_all[:, observed_vars]
            original_graph, original_vals = modify_dict_get_graph_and_link_vals(links)
            dataframe = pp.DataFrame(data)

            # save data to file
            # filename = os.path.abspath("./../../../test.dat")
            # fileobj = open(filename, mode='wb')
            # off = np.array(data, dtype=np.float32)
            # off.tofile(fileobj)
            # fileobj.close()

            # plot data
            if plot_data:
                tp.plot_timeseries(dataframe_all, figsize=(15, 5));
                plt.show()

            # plot original DAG
            if verbosity > 0:
                tp.plot_graph(
                    val_matrix=original_vals,  # original_vals None
                    link_matrix=original_graph,
                    var_names=range(N_all),
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
                #     var_names=range(N_all),
                #     link_colorbar_label='MCI',
                # )
                # plt.show()
            return dataframe, links, observed_vars, original_graph


        else:
            print("Trial %d: Not a stationary model" % ir)
            model_seed += 10000
            if ir > 998:
                raise ValueError('datagenerating process unstable')


def compute_oracle_pag(links, observed_vars, tau_max):
    """
    Compute the oracle PAG, given links and observed_vars and tau_max
    returns: oracle_pag
    """
    oracle_graph = utilities.get_oracle_pag_from_dag(links, observed_vars=observed_vars, tau_max=tau_max,
                                                     verbosity=verbosity)[1]
    if verbosity > 0:
        # plot oracle PAG

        # plot oralce PAG
        tp.plot_graph(
            val_matrix=None,
            link_matrix=oracle_graph,
            var_names=observed_vars,
            link_colorbar_label='cross-MCI',
            node_colorbar_label='auto-MCI',
            figsize=(10, 6),
        )
        plt.show()
        # Plot time series graph
        tp.plot_time_series_graph(
            figsize=(12, 8),
            val_matrix=None,
            link_matrix=oracle_graph,
            var_names=observed_vars,
            link_colorbar_label='MCI',
        )
        plt.show()

        print("True Links")
        for j in links:
            print(j, links[j])
        print("observed_vars = ", observed_vars)
        print("True PAG")
        if tau_max > 0:
            for lag in range(tau_max + 1):
                print(oracle_graph[:, :, lag])
        else:
            print(oracle_graph.squeeze())
    return oracle_graph


def calculate(para_setup):
    """
    Main function to run the experiment, given para_setup

    returns: original_graph, oracle_graph, val_min, max_cardinality,

    calls:
    1. parses input parameters
    2. calls generate_dataframe
    3. calls compute_oracle_pag
    4. calls LPCMCI to get graph and values

    """
    para_setup_string, sam = para_setup

    paras = para_setup_string.split('-')
    paras = [w.replace("'", "") for w in paras]

    model = 'random_lineargaussian'
    N = 8
    n_links = 8
    min_coeff = 0.2
    coeff = 0.5
    auto = 0.5  # auto-dependency (auto-correlation) 0.5
    contemp_fraction = 0.6
    frac_unobserved = 0.3
    max_true_lag = 1
    T = 500

    ci_test = 'parr_corr'
    method = 'lpcmci_nprelim4'
    pc_alpha = 0.26
    tau_max = 1

    #############################################
    ##  Data
    #############################################

    dataframe, links, observed_vars, original_graph = generate_dataframe(model, coeff, min_coeff, auto, sam, N,
                                                                         frac_unobserved,
                                                                         n_links, max_true_lag, T, contemp_fraction)

    # dataframe, links, observed_vars = generate_fixed_data()

    #############################################
    ##  Methods
    #############################################
    oracle_graph = compute_oracle_pag(links, observed_vars, tau_max)

    computation_time_start = time.time()

    lpcmci = LPCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(significance='analytic', recycle_residuals=True)
    )

    lpcmci.run_lpcmci(
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        max_p_non_ancestral=3,  # max cardinality of conditioning set, in the second removal phase
        n_preliminary_iterations=10,
        verbosity=verbosity)

    graph = lpcmci.graph
    l = ['-->', '', '<--', '', 'o->', '', '<-o', '', '<->', '']
    random_edgemark_graph = [[[rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)]],
              [[rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)]],
              [[rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)]],
              [[rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)]],
              [[rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)]],
              [[rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)]],
              [[rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)]],
              [[rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)],
               [rd.choice(l), rd.choice(l)], [rd.choice(l), rd.choice(l)]]]
    # graph = np.asarray(random_edgemark_graph)
    # print('\ngraph!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n', graph, '\n\n')

    val_min = lpcmci.val_min_matrix
    max_cardinality = lpcmci.cardinality_matrix

    # pcmci = PCMCI(
    #     dataframe=dataframe,
    #     cond_ind_test=ParCorr(significance='analytic'),
    #     verbosity=1)
    #
    # results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
    # q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh',
    #                                        exclude_contemporaneous=False)
    # link_matrix = results['graph']
    #
    # graph = link_matrix
    # val_min = results['val_matrix']
    # max_cardinality = None

    computation_time_end = time.time()
    computation_time = computation_time_end - computation_time_start

    # plot predicted PAG
    if verbosity > 0:
        tp.plot_graph(
            val_matrix=val_min,
            link_matrix=graph,
            var_names=observed_vars,
            link_colorbar_label='cross-MCI',
            node_colorbar_label='auto-MCI',
            figsize=(10, 6),
        )
        plt.show()
        # Plot time series graph
        tp.plot_time_series_graph(
            figsize=(12, 8),
            val_matrix=val_min,
            link_matrix=graph,
            var_names=observed_vars,
            link_colorbar_label='MCI',
        )
        plt.show()

        # reduced links
        # reduced_val_min = val_min
        # reduced_graph = graph
        # reduced_val_min[abs(reduced_val_min) < remove_link_threshold] = 0  # set values below threshold to zero
        # reduced_graph[abs(reduced_val_min) < remove_link_threshold] = ""  # set values below threshold to zero
        # tp.plot_graph(
        #     val_matrix=reduced_val_min,
        #     link_matrix=reduced_graph,
        #     var_names=observed_vars,
        #     link_colorbar_label='cross-MCI',
        #     node_colorbar_label='auto-MCI',
        #     figsize=(10, 6),
        # )
        # plt.show()
        # # Plot time series graph
        # tp.plot_time_series_graph(
        #     figsize=(12, 8),
        #     val_matrix=reduced_val_min,
        #     link_matrix=reduced_graph,
        #     var_names=observed_vars,
        #     link_colorbar_label='MCI',
        # )
        # plt.show()

    return {
        'original_graph': original_graph,
        'oracle_graph': oracle_graph,
        'val_min': val_min,
        'max_cardinality': max_cardinality,

        # Method results
        'computation_time': computation_time,
        'graph': graph,
    }


if __name__ == '__main__':
    """
    calls calcualte()
    
    """

    all_configs = dict([(conf, {'results': {},
                                "graphs": {},
                                "val_min": {},
                                "max_cardinality": {},

                                "oracle_graph": {},
                                "original_graph": {},
                                "computation_time": {}, }) for conf in config_list])

    job_list = [(conf, i) for i in range(samples) for conf in config_list]

    num_tasks = len(job_list)

    for config_sam in job_list:
        config, sample = config_sam
        print("Experiment %s - Realization %d" % (config, sample))
        ##################
        ### calculate ###
        ##################
        all_configs[config]['results'][sample] = calculate(config_sam)

    print("\nsaving all configs...")

    for conf in list(all_configs.keys()):
        all_configs[conf]['graphs'] = np.zeros((samples,) + all_configs[conf]['results'][0]['graph'].shape, dtype='<U3')
        all_configs[conf]['oracle_graphs'] = np.zeros(
            (samples,) + all_configs[conf]['results'][0]['oracle_graph'].shape,
            dtype='<U3')
        all_configs[conf]['original_graphs'] = np.zeros(
            (samples,) + all_configs[conf]['results'][0]['original_graph'].shape,
            dtype='<U3')
        all_configs[conf]['val_min'] = np.zeros((samples,) + all_configs[conf]['results'][0]['val_min'].shape)
        all_configs[conf]['max_cardinality'] = np.zeros(
            (samples,) + all_configs[conf]['results'][0]['max_cardinality'].shape)
        all_configs[conf]['computation_time'] = []

        for i in list(all_configs[conf]['results'].keys()):
            all_configs[conf]['graphs'][i] = all_configs[conf]['results'][i]['graph']
            all_configs[conf]['original_graphs'][i] = all_configs[conf]['results'][i]['original_graph']
            all_configs[conf]['oracle_graphs'][i] = all_configs[conf]['results'][i]['oracle_graph']
            all_configs[conf]['val_min'][i] = all_configs[conf]['results'][i]['val_min']
            all_configs[conf]['max_cardinality'][i] = all_configs[conf]['results'][i]['max_cardinality']

            all_configs[conf]['computation_time'].append(all_configs[conf]['results'][i]['computation_time'])

        # Save all results
        file_name = folder_name + '%s' % (conf)

        # Compute and save metrics in separate (smaller) file
        metrics = metrics_mod.get_evaluation(results=all_configs[conf])
        for metric in metrics:
            if metric != 'computation_time':
                print(f"{metric:30s} {metrics[metric][0]: 1.2f} +/-{metrics[metric][1]: 1.2f} ")
            else:
                print(
                    f"{metric:30s} {metrics[metric][0]: 1.2f} +/-[{metrics[metric][1][0]: 1.2f}, {metrics[metric][1][1]: 1.2f}]")
        # chr:
        f1_score_adjacency = utilities.compute_f1_score(metrics['adj_anylink_precision'][0],
                                                        metrics['adj_anylink_recall'][0])
        f1_score_edgemark = utilities.compute_f1_score(metrics['edgemarks_anylink_precision'][0],
                                                       metrics['edgemarks_anylink_recall'][0])
        print('f1_score_adjacency:', f1_score_adjacency, '\nf1_score_edgemark:', f1_score_edgemark)

        print("Metrics dump ", file_name.replace("'", "").replace('"', '') + '_metrics.dat')
        file = open(file_name.replace("'", "").replace('"', '') + '_metrics.dat', 'wb')
        pickle.dump(metrics, file, protocol=-1)
        file.close()

        del all_configs[conf]['results']

        # Also save raw results
        print("dump ", file_name.replace("'", "").replace('"', '') + '.dat')
        file = open(file_name.replace("'", "").replace('"', '') + '.dat', 'wb')
        pickle.dump(all_configs[conf], file, protocol=-1)
        file.close()

    time_end = time.time()
    print('Run time in hours ', (time_end - time_start) / 3600.)
