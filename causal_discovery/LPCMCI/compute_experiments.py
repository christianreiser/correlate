import math
import pickle
import time

import numpy as np
# Imports from tigramite package available on https://github.com/jakobrunge/tigramite
import tigramite.data_processing as pp
from matplotlib import pyplot, pyplot as plt
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr, GPDC, CMIknn

# Imports from code inside directory
import generate_data_mod as mod
import metrics_mod
import utilities as utilities
from discG2 import DiscG2
from lpcmci import LPCMCI
from simulate_discrete_scm import discretized_scp

# Directory to save results
folder_name = "results/"

# Arguments passed via command line
# arg = sys.argv
# 'model-N-L-min_coeff-max_coeff-autocorr-frac_contemp_links-frac_unobserved-max_true_lag-time_series_length-CI_test-method-alpha_level-tau_max'

# original
arg = [0, 1, 2, 'random_lineargaussian-3-3-0.2-0.8-0.9-0.3-0.3-3-100-par_corr-lpcmci_nprelim4-0.05-5']

# easy and correct
# arg = [0, 1, 0, 'random_lineargaussian-2-2-0.5-0.9-0.9-0.3-0.3-1-1000-par_corr-lpcmci_nprelim4-0.05-1']

# close to real
arg = [0, 1, 3, 'random_lineargaussian-9-9-0.1-0.45-0.6-0.6-0.3-1-510-par_corr-lpcmci_nprelim4-0.26-1'] # alpha=0.26

remove_link_threshold = 0.11
samples = int(arg[1])  # int number of time series realizations to generate
verbosity = int(arg[2])  # verbosity
config_list = list(arg)[3:]  # string that identifies a particular experiment consisting of a model and method.
num_configs = len(config_list)

time_start = time.time()

if verbosity > 1:
    plot_data = True
else:
    plot_data = False


def modify_dict_get_graph_and_link_vals(my_dict):
    """
    outputs:
    1. new dict with link
    2. link graph
    3. val graph

    input:
    dict with format s.th. like
    my_dict = {0: [((0, -1), 0.85, 'removeme'),
                   ((1, 0), -0.5, 'removeme'),
                   ((2, -1), 0.7, 'removeme')],
               1: [((1, -1), 0.8, 'removeme'),
                   ((2, 0), 0.7, 'removeme')],
               2: [((2, -1), 0.9, 'removeme')],
               3: [((3, -2), 0.8, 'removeme'),
                   ((0, -3), 0.4, 'removeme')]}
    """
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

    print('links:', my_dict)

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
    return my_dict, graph, val


def calculate(para_setup):
    para_setup_string, sam = para_setup

    paras = para_setup_string.split('-')
    paras = [w.replace("'", "") for w in paras]

    model = str(paras[0])  # e.g. random_lineargaussian
    N = int(paras[1])  # 3
    n_links = int(paras[2])  # 3
    min_coeff = float(paras[3])  # 0.2
    coeff = float(paras[4])  # 0.8
    auto = float(paras[5])  # auto-dependency (auto-correlation) 0.9
    contemp_fraction = float(paras[6])  # 0.3
    frac_unobserved = float(paras[7])  # 0.3
    max_true_lag = int(paras[8])  # 1
    T = int(paras[9])  # 100

    ci_test = str(paras[10])  # parr_corr
    method = str(paras[11])  # lpcmci_nprelim4
    pc_alpha = float(paras[12])  # 0.05
    tau_max = int(paras[13])  # 5

    #############################################
    ##  Data
    #############################################

    def lin_f(x):
        return x

    def f2(x):
        return (x + 5. * x ** 2 * np.exp(-x ** 2 / 20.))

    if 'random' in model:
        if 'lineargaussian' in model:
            coupling_funcs = [lin_f]

            noise_types = ['gaussian']  # , 'weibull', 'uniform']
            noise_sigma = (0.5, 2)

        # elif 'nonlinearmixed' in model:
        #
        #     coupling_funcs = [lin_f, f2]
        #
        #     noise_types = ['gaussian', 'gaussian', 'weibull']
        #     noise_sigma = (0.5, 2)

        if coeff < min_coeff:  # correct coeff if to small
            min_coeff = coeff
        couplings = list(np.arange(min_coeff, coeff + 0.1, 0.1))  # coupling strength
        couplings += [-c for c in couplings]  # add negative coupling strength

        auto_deps = list(np.arange(max(0., auto - 0.3), auto + 0.01, 0.05))  # auto-correlations

        # Models may be non-stationary. Hence, we iterate over a number of seeds
        # to find a stationary one regarding network topology, noises, etc
        if verbosity > 999:
            model_seed = verbosity - 1000
        else:
            model_seed = sam

        for ir in range(1000):
            random_state = np.random.RandomState(model_seed)

            N_all = math.floor((N / (1. - frac_unobserved)))  # 4
            n_links_all = math.ceil(n_links / N * N_all)  # 4
            observed_vars = np.sort(random_state.choice(range(N_all),  # [1,2,3]
                                                        size=math.ceil((1. - frac_unobserved) * N_all),
                                                        replace=False)).tolist()

            links = mod.generate_random_contemp_model(
                N=N_all, L=n_links_all,
                coupling_coeffs=couplings,
                coupling_funcs=coupling_funcs,
                auto_coeffs=auto_deps,
                tau_max=max_true_lag,
                contemp_fraction=contemp_fraction,
                # num_trials=1000,  
                random_state=random_state)

            class NoiseModel:
                def __init__(self, sigma=1):
                    self.sigma = sigma

                def gaussian(self, T):
                    # Get zero-mean unit variance gaussian distribution
                    return self.sigma * random_state.randn(T)

                # def weibull(self, T):
                #     # Get zero-mean sigma variance weibull distribution
                #     a = 2
                #     mean = scipy.special.gamma(1. / a + 1)
                #     variance = scipy.special.gamma(2. / a + 1) - scipy.special.gamma(1. / a + 1) ** 2
                #     return self.sigma * (random_state.weibull(a=a, size=T) - mean) / np.sqrt(variance)
                #
                # def uniform(self, T):
                #     # Get zero-mean sigma variance uniform distribution
                #     mean = 0.5
                #     variance = 1. / 12.
                #     return self.sigma * (random_state.uniform(size=T) - mean) / np.sqrt(variance)

            noises = []
            for j in links:
                noise_type = random_state.choice(noise_types)  # gaussian
                sigma = noise_sigma[0] + (noise_sigma[1] - noise_sigma[0]) * random_state.rand()  # 2,1.2,1,7
                noises.append(getattr(NoiseModel(sigma), noise_type))

            if 'discretebinom' in model:  # False
                if 'binom2' in model:
                    n_binom = 2
                elif 'binom4' in model:
                    n_binom = 4

                data_all_check, nonstationary = discretized_scp(links=links, T=T + 10000,
                                                                n_binom=n_binom, random_state=random_state)
            else:  # yes
                data_all_check, nonstationary = mod.generate_nonlinear_contemp_timeseries(
                    links=links, T=T + 10000, noises=noises, random_state=random_state)

            # If the model is stationary, break the loop
            if not nonstationary:
                data_all = data_all_check[:T]
                data = data_all[:, observed_vars]
                break
            else:
                print("Trial %d: Not a stationary model" % ir)
                model_seed += 10000
    # elif model == 'autobidirected':
    #     if verbosity > 999:
    #         model_seed = verbosity - 1000
    #     else:
    #         model_seed = sam
    #
    #     random_state = np.random.RandomState(model_seed)
    #
    #     links = {
    #         0: [((0, -1), auto, lin_f), ((1, -1), coeff, lin_f)],
    #         1: [],
    #         2: [((2, -1), auto, lin_f), ((1, -1), coeff, lin_f)],
    #         3: [((3, -1), auto, lin_f), ((2, -1), min_coeff, lin_f)],
    #     }
    #     observed_vars = [0, 2, 3]
    #
    #     noises = [random_state.randn for j in range(len(links))]
    #
    #     data_all, nonstationary = mod.generate_nonlinear_contemp_timeseries(
    #         links=links, T=T, noises=noises, random_state=random_state)
    #     data = data_all[:, observed_vars]
    else:
        raise ValueError("model %s not known" % model)

    if nonstationary:
        raise ValueError("No stationary model found: %s" % model)

    links_dict_clean, original_graph, original_vals = modify_dict_get_graph_and_link_vals(links)

    true_graph = utilities.get_pag_from_dag(links, observed_vars=observed_vars,
                                            tau_max=tau_max, verbosity=verbosity)[1]

    if verbosity > 0:
        print("True Links")
        for j in links:
            print(j, links[j])
        print("observed_vars = ", observed_vars)
        print("True PAG")
        if tau_max > 0:
            for lag in range(tau_max + 1):
                print(true_graph[:, :, lag])
        else:
            print(true_graph.squeeze())

    if plot_data:
        print("PLOTTING")
        for j in range(N):
            # ax = fig.add_subplot(N,1,j+1)
            pyplot.plot(data[:, j])

        pyplot.show()

    computation_time_start = time.time()

    dataframe = pp.DataFrame(data)

    #############################################
    ##  Methods
    #############################################

    # Specify conditional independence test object
    if ci_test == 'par_corr':
        cond_ind_test = ParCorr(
            significance='analytic',
            recycle_residuals=True)
    elif ci_test == 'cmi_knn':
        cond_ind_test = CMIknn(knn=0.1,
                               sig_samples=500,
                               sig_blocklength=1)
    elif ci_test == 'gp_dc':
        cond_ind_test = GPDC(
            recycle_residuals=True)
    elif ci_test == 'discg2':
        cond_ind_test = DiscG2()
    else:
        raise ValueError("CI test not recognized.")

    if 'lpcmci' in method:
        method_paras = method.split('_')
        n_preliminary_iterations = int(method_paras[1][7:])

        if 'prelimonly' in method:
            prelim_only = True
        else:
            prelim_only = False

        lpcmci = LPCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test)

        lpcmci.run_lpcmci(
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            max_p_non_ancestral=3,
            n_preliminary_iterations=n_preliminary_iterations,
            prelim_only=prelim_only,
            verbosity=verbosity)

        graph = lpcmci.graph
        val_min = lpcmci.val_min_matrix
        max_cardinality = lpcmci.cardinality_matrix
    else:
        raise ValueError("%s not implemented." % method)

    computation_time_end = time.time()
    computation_time = computation_time_end - computation_time_start

    if verbosity > 1:
        # plot original DAG
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
        tp.plot_time_series_graph(
            figsize=(12, 8),
            val_matrix=original_vals,  # original_vals None
            link_matrix=original_graph,
            var_names=range(N_all),
            link_colorbar_label='MCI',
        )
        plt.show()

        # plot true PAG
        tp.plot_graph(
            val_matrix=None,
            link_matrix=true_graph,
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
            link_matrix=true_graph,
            var_names=observed_vars,
            link_colorbar_label='MCI',
        )
        plt.show()

        # val_min[abs(val_min) < remove_link_threshold] = 0  # set values below threshold to zero
        # graph[abs(val_min) < remove_link_threshold] = ""  # set values below threshold to zero

        # plot found PAG
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

    return {
        'true_graph': true_graph,
        'val_min': val_min,
        'max_cardinality': max_cardinality,

        # Method results
        'computation_time': computation_time,
        'graph': graph,
    }


if __name__ == '__main__':

    all_configs = dict([(conf, {'results': {},
                                "graphs": {},
                                "val_min": {},
                                "max_cardinality": {},

                                "true_graph": {},
                                "computation_time": {}, }) for conf in config_list])

    job_list = [(conf, i) for i in range(samples) for conf in config_list]

    num_tasks = len(job_list)

    for config_sam in job_list:
        config, sample = config_sam
        print("Experiment %s - Realization %d" % (config, sample))
        all_configs[config]['results'][sample] = calculate(config_sam)

    print("\nsaving all configs...")

    for conf in list(all_configs.keys()):
        all_configs[conf]['graphs'] = np.zeros((samples,) + all_configs[conf]['results'][0]['graph'].shape, dtype='<U3')
        all_configs[conf]['true_graphs'] = np.zeros((samples,) + all_configs[conf]['results'][0]['true_graph'].shape,
                                                    dtype='<U3')
        all_configs[conf]['val_min'] = np.zeros((samples,) + all_configs[conf]['results'][0]['val_min'].shape)
        all_configs[conf]['max_cardinality'] = np.zeros(
            (samples,) + all_configs[conf]['results'][0]['max_cardinality'].shape)
        all_configs[conf]['computation_time'] = []

        for i in list(all_configs[conf]['results'].keys()):
            all_configs[conf]['graphs'][i] = all_configs[conf]['results'][i]['graph']
            all_configs[conf]['true_graphs'][i] = all_configs[conf]['results'][i]['true_graph']
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
        f1_score_anylink = utilities.compute_f1_score(metrics['adj_anylink_precision'][0],
                                                      metrics['adj_anylink_recall'][0])
        f1_score_edgemark = utilities.compute_f1_score(metrics['edgemarks_anylink_precision'][0],
                                                       metrics['edgemarks_anylink_recall'][0])
        print('f1_score_anylink:', f1_score_anylink, '\nf1_score_edgemark:', f1_score_edgemark)

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
