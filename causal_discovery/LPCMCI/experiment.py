import numpy as np
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr
from causal_discovery.LPCMCI import utilities, metrics_mod
from causal_discovery.LPCMCI.lpcmci import LPCMCI
from causal_discovery.LPCMCI.utilities import get_pag_from_dag
from config import verbosity, causal_discovery_on, tau_max


def calculate():
    links = {0: [((0, -1), 0.4), ((1, -1), 1)],
             1: [((1, -1), 0.4), ((3, -1), +1)],
             2: [((2, -1), 0.4), ((3, -1), 1)],
             3: [((3, -1), -0.4), ((4, -1), +1)],
             4: [((4, -1), 0.4)],
             }
    T = 1000  # time series length

    # np.random.seed(41)  # Fix random seed
    data, true_parents_neighbors = pp.var_process(links, T=T, use='inv_inno_cov')  # inv_inno_cov no_noise
    # data = np.delete(data, 3, 1) # hide X3

    # Initialize dataframe object, specify time axis and variable names
    var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$', r'$X^4$']
    # var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^4$']
    dataframe = pp.DataFrame(data,
                             datatime=np.arange(len(data)),
                             var_names=var_names)

    tp.plot_timeseries(dataframe)
    plt.show()

    observed_vars = [0, 1, 2, 4]
    true_graph = get_pag_from_dag(links, observed_vars=observed_vars,
                                             tau_max=tau_max, verbosity=verbosity)

    cond_ind_test = ParCorr(
        significance='analytic',
        recycle_residuals=True)

    lpcmci = LPCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test)

    n_preliminary_iterations = 2
    prelim_only = False

    lpcmcires = lpcmci.run_lpcmci(
        tau_max=tau_max,  # maximum considered time lag tau_max
        pc_alpha=0.5,  # significance level \alpha of conditional independence tests
        # max_p_non_ancestral=3,  # Restricts all conditional independence tests in the second removal phase
        n_preliminary_iterations=n_preliminary_iterations,  # In the paper this corresponds to the 'k' in LPCMCI(k)
        prelim_only=prelim_only,  # If True, stop after the preliminary phase. For detailed performance analysis
        verbosity=verbosity)

    graph = lpcmci.graph
    val_min = lpcmci.val_min_matrix
    max_cardinality = lpcmci.cardinality_matrix

    # chr: remove weak links
    val_min[val_min < 0.1] = 0  # set values below threshold to zero
    graph[val_min < 0.1] = ""  # set values below threshold to zero

    # plot
    tp.plot_graph(
        val_matrix=val_min,
        link_matrix=graph,
        var_names=var_names,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI',
        figsize=(10, 6),
    )

    plt.show()
    #
    # # Plot time series graph
    # tp.plot_time_series_graph(
    #     figsize=(12, 8),
    #     val_matrix=results['val_matrix'],
    #     link_matrix=link_matrix,
    #     var_names=var_names,
    #     link_colorbar_label='MCI',
    # )
    # plt.show()

    return {
        'true_graph': true_graph,
        'val_min': val_min,
        'max_cardinality': max_cardinality,

        # Method results
        'graph': graph,
    }


def causal_discovery():
    if causal_discovery_on:
        # select columns
        # df = df[['HumidInMax()', 'Mood', 'HeartPoints', 'CO2Median()', 'NoiseMax()', 'VitaminDSup', 'PressOutMin()',
        # 'BodyWeight']] #
        # df.reset_index(level=0, inplace=True)
        #
        # df = remove_nan_seq_from_top_and_bot(df)
        # # df = non_contemporary_time_series_generation(df)  # todo, how to automate on and off
        # df = df.drop(['Date'], axis=1)  # drop date col
        #
        # # # standardize data
        # df -= df.mean(axis=0)
        # df /= df.std(axis=0)
        #
        # var_names = df.columns
        # dataframe = pp.DataFrame(df.values, datatime=np.arange(len(df)),
        #                          var_names=var_names)

        n_time_series_realizations = 20
        config_list = 'random_lineargaussian-3-3-0.2-0.8-0.9-0.3-0.3-3-100-par_corr-lpcmci_nprelim2-0.05-5'

        all_configs = dict([(conf, {'results': {},
                                    "graphs": {},
                                    "val_min": {},
                                    "max_cardinality": {},

                                    "true_graph": {},
                                    "computation_time": {}, }) for conf in config_list])

        job_list = [(conf, i) for i in range(n_time_series_realizations) for conf in config_list]
        for config_sam in job_list:
            config, sample = config_sam
            print("Experiment %s - Realization %d" % (config, sample))
            all_configs[config]['results'][sample] = calculate()

        print("\nsaving all configs...")

        for conf in list(all_configs.keys()):
            all_configs[conf]['graphs'] = np.zeros(
                (n_time_series_realizations,) + all_configs[conf]['results'][0]['graph'].shape,
                dtype='<U3')
            all_configs[conf]['true_graphs'] = np.zeros(
                (n_time_series_realizations,) + all_configs[conf]['results'][0]['true_graph'].shape,
                dtype='<U3')
            all_configs[conf]['val_min'] = np.zeros(
                (n_time_series_realizations,) + all_configs[conf]['results'][0]['val_min'].shape)
            all_configs[conf]['max_cardinality'] = np.zeros(
                (n_time_series_realizations,) + all_configs[conf]['results'][0]['max_cardinality'].shape)
            all_configs[conf]['computation_time'] = []

            for i in list(all_configs[conf]['results'].keys()):
                all_configs[conf]['graphs'][i] = all_configs[conf]['results'][i]['graph']
                all_configs[conf]['true_graphs'][i] = all_configs[conf]['results'][i]['true_graph']
                all_configs[conf]['val_min'][i] = all_configs[conf]['results'][i]['val_min']
                all_configs[conf]['max_cardinality'][i] = all_configs[conf]['results'][i]['max_cardinality']

                all_configs[conf]['computation_time'].append(all_configs[conf]['results'][i]['computation_time'])

        # Compute and save metrics in separate (smaller) file
        metrics = metrics_mod.get_evaluation(results=all_configs[conf])
causal_discovery()