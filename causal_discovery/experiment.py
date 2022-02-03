import numpy as np
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI

from causal_discovery.LPCMCI.lpcmci import LPCMCI
from causal_discovery.preprocessing import remove_nan_seq_from_top_and_bot
from config import verbosity, alpha_level, causal_discovery_on, pc_alpha


def causal_discovery(df, tau_max):
    if causal_discovery_on:
        # select columns
        # df = df[['HumidInMax()', 'Mood', 'HeartPoints', 'CO2Median()', 'NoiseMax()', 'VitaminDSup', 'PressOutMin()', 'BodyWeight']] #
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

        np.random.seed(42)  # Fix random seed

        links_coeffs = {0: [((0, -1), 0.9), ((1, -1), 0.9)],
                        1: [((1, -1), 0.8), ((2, -2), 0.8)],
                        2: [((2, -1), 0.7), ((3, -3), 0.7)],
                        3: [((3, -1), 0.6), ((4, -4), 0.6)],
                        4: [((4, -1), 0.5), ((4, -5), -0.4)],
                        }
        T = 1000  # time series length
        data, true_parents_neighbors = pp.var_process(links_coeffs, T=T)
        T, N = data.shape

        # Initialize dataframe object, specify time axis and variable names
        var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$', r'$X^4$']
        dataframe = pp.DataFrame(data,
                                 datatime=np.arange(len(data)),
                                 var_names=var_names)


        tp.plot_timeseries(dataframe)
        plt.show()

        cond_ind_test = ParCorr(
            significance='analytic',
            recycle_residuals=True)

        # pcmci = PCMCI(
        #     dataframe=dataframe,
        #     cond_ind_test=parcorr,
        #     verbosity=verbosity)
        #
        # """
        #  run_bivci implements a bivariate, lagged conditional independence test (similar to bivariate Granger
        #  causality, but lag-specific). This can help to identify which maximal time lag tau_max to choose.
        #  Another option would be to plot get_lagged_dependencies, but large autocorrelation will inflate lag peaks
        #  (see https://journals.ametsoc.org/doi/full/10.1175/JCLI-D-13-00159.1) and run_bivci at least conditions out
        #  some part of the autocorrelation.
        #  """
        # correlations = pcmci.run_bivci(tau_max=tau_max, val_only=True)['val_matrix']  # get_lagged_dependencies
        # tp.plot_lagfuncs(val_matrix=correlations,
        #                  setup_args={'var_names': var_names, 'x_base': 5, 'y_base': .5})
        # plt.show()
        #
        # tau_max = reduce_tau_max(correlations)
        #
        # pcmci.verbosity = verbosity
        # # results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None)
        #
        # """
        # Since the dependencies peak maximally at a lag of around 3, we choose tau_max=3 for PCMCIplus. This choice may,
        # however, stronly depend on expert knowledge of the system. Obviously, for contemporaneous causal discovery, we leave
        # the default tau_min=0. The other main parameter is pc_alpha which sets the significance level for all tests in
        # PCMCIplus. This is in contrast to PCMCI where pc_alpha only controls the significance tests in the
        # condition-selection phase, not in the MCI tests. Also for PCMCIplus there is an automatic procedure (like for PCMCI)
        #  to choose the optimal value. If a list or None is passed for pc_alpha, the significance level is optimized for
        #  every graph across the given pc_alpha values using the score computed in
        #  cond_ind_test.get_model_selection_criterion(). Since PCMCIplus outputs not a DAG, but an equivalence class of DAGs,
        #  first one member is of this class is computed and then the score is computed as the average over all models fits
        #  for each variable. The score is the same for all members of the class.
        # Here we set it to pc_alpha=0.01. In applications a number of different values should be tested and results
        # transparently discussed.
        # It is instructive to set verbosity=2 and understand the output of PCMCIplus, after reading the paper and the
        # pseudo-code. In the output contemporaneous adjacencies which are not oriented are marked by o--o and already
        # oriented adjacencies by -->.
        # """
        # # pcmci.verbosity = 2
        # results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
        #
        # q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=tau_max, fdr_method='fdr_bh',
        #                                        exclude_contemporaneous=False)



        lpcmci = LPCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test)

        n_preliminary_iterations = 2
        prelim_only = False

        lpcmcires = lpcmci.run_lpcmci(
                    tau_max=tau_max,
                    pc_alpha=0.05, #pc_alpha,
                    max_p_non_ancestral=3,
                    n_preliminary_iterations=n_preliminary_iterations,
                    prelim_only=prelim_only,
                    verbosity=verbosity)

        graph = lpcmci.graph
        val_min = lpcmci.val_min_matrix
        max_cardinality = lpcmci.cardinality_matrix



        # pcmci.print_significant_links(
        #     p_matrix=results['p_matrix'],
        #     q_matrix=q_matrix,
        #     val_matrix=results['val_matrix'],
        #     alpha_level=alpha_level)
        #
        # # link_matrix = pcmci.return_significant_links(pq_matrix=q_matrix,
        # #                                              val_matrix=results['val_matrix'], alpha_level=alpha_level)[
        # #     'link_matrix']
        #
        # link_matrix = results['graph']
        #
        #
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


