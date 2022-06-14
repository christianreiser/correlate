import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI

from causal_discovery.LPCMCI.lpcmci import LPCMCI
from causal_discovery.preprocessing import remove_nan_seq_from_top_and_bot
from config import verbosity, causal_discovery_on, tau_max, pc_alpha, private_folder_path, remove_link_threshold, \
    LPCMCI_or_PCMCI


# def calculate():
#     links = {0: [((0, -1), 0.4), ((1, -1), 1)],
#              1: [((1, -1), 0.4), ((3, -1), +1)],
#              2: [((2, -1), 0.4), ((3, -1), 1)],
#              3: [((3, -1), -0.4), ((4, -1), +1)],
#              4: [((4, -1), 0.4)],
#              }
#     T = 1000  # time series length
#
#     # np.random.seed(41)  # Fix random seed
#     data, true_parents_neighbors = pp.var_process(links, T=T, use='inv_inno_cov')  # inv_inno_cov no_noise
#     # data = np.delete(data, 3, 1) # hide X3
#
#     # Initialize dataframe object, specify time axis and variable names
#     var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$', r'$X^4$']
#     # var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^4$']
#     dataframe = pp.DataFrame(data,
#                              datatime=np.arange(len(data)),
#                              var_names=var_names)
#
#     tp.plot_timeseries(dataframe)
#     plt.show()
#
#
#     cond_ind_test = ParCorr(
#         significance='analytic',
#         recycle_residuals=True)
#
#     lpcmci = LPCMCI(
#         dataframe=dataframe,
#         cond_ind_test=cond_ind_test)
#
#     n_preliminary_iterations = 2
#     prelim_only = False
#
#     lpcmcires = lpcmci.run_lpcmci(
#         tau_max=tau_max,  # maximum considered time lag tau_max
#         pc_alpha=0.5,  # significance level \alpha of conditional independence tests
#         # max_p_non_ancestral=3,  # Restricts all conditional independence tests in the second removal phase
#         n_preliminary_iterations=n_preliminary_iterations,  # In the paper this corresponds to the 'k' in LPCMCI(k)
#         prelim_only=prelim_only,  # If True, stop after the preliminary phase. For detailed performance analysis
#         verbosity=verbosity)
#
#     graph = lpcmci.graph
#     val_min = lpcmci.val_min_matrix
#
#     # chr: remove weak links
#     val_min[val_min < 0.1] = 0  # set values below threshold to zero
#     graph[val_min < 0.1] = ""  # set values below threshold to zero
#
#     # plot
#     tp.plot_graph(
#         val_matrix=val_min,
#         link_matrix=graph,
#         var_names=var_names,
#         link_colorbar_label='cross-MCI',
#         node_colorbar_label='auto-MCI',
#         figsize=(10, 6),
#     )
#
#     plt.show()
#     #
#     # # Plot time series graph
#     # tp.plot_time_series_graph(
#     #     figsize=(12, 8),
#     #     val_matrix=results['val_matrix'],
#     #     link_matrix=link_matrix,
#     #     var_names=var_names,
#     #     link_colorbar_label='MCI',
#     # )
#     # plt.show()
#
#     return {
#         # Method results
#         'graph': graph,
#     }

# function that saves val_min, graph, and var_names to a file
def save_results(val_min, graph, var_names, name_extension):
    np.save(str(private_folder_path) + 'val_min_'+str(name_extension), val_min)
    np.save(str(private_folder_path) + 'graph_'+str(name_extension), graph)
    np.save(str(private_folder_path) + 'var_names_'+str(name_extension), var_names)


def causal_discovery(df):
    if causal_discovery_on:

        """get non_zero_indices"""
        # non_zero_inices = pd.read_csv(str(private_folder_path) + 'results.csv', index_col=0)
        # # of non_zero_inices get column called 'ref_coeff_widestk=5'
        # non_zero_inices = non_zero_inices.loc[:, 'reg_coeff_widestk=5']
        # # drop all rows with 0 in non_zero_inices
        # non_zero_inices = non_zero_inices[non_zero_inices != 0]
        # # detete all rows with nans in non_zero_inices
        # non_zero_inices = non_zero_inices.dropna().index
        # TODO: automatic non_zero_inices doesn't work yet below is hardcoded
        # non_zero_inices = ['Mood', 'HumidInMax()', 'NoiseMax()', 'HeartPoints', 'Steps']

        # select columns
        # df = df[non_zero_inices]
        # df.reset_index(level=0, inplace=True)

        # df = remove_nan_seq_from_top_and_bot(df)
        # df = non_contemporary_time_series_generation(df)  # todo, how to automate on and off
        # df = df.drop(['Date'], axis=1)  # drop date col

        # # standardize data
        df -= df.mean(axis=0)
        df /= df.std(axis=0)

        var_names = df.columns
        dataframe = pp.DataFrame(df.values, datatime=np.arange(len(df)),
                                 var_names=var_names)

        if LPCMCI_or_PCMCI:
            lpcmci = LPCMCI(
                dataframe=dataframe,
                cond_ind_test=ParCorr(
                    significance='analytic',
                    recycle_residuals=True))

            lpcmci.run_lpcmci(
                tau_max=tau_max,
                pc_alpha=pc_alpha,
                max_p_non_ancestral=3,
                n_preliminary_iterations=4,
                prelim_only=False,
                verbosity=verbosity)

            graph = lpcmci.graph
            val_min = lpcmci.val_min_matrix

        else:
            """pcmci"""
            pcmci = PCMCI(
                dataframe=dataframe,
                cond_ind_test=ParCorr(significance='analytic'),
                verbosity=1)

            results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
            q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh',
                                                   exclude_contemporaneous=False)

            graph = results['graph']
            val_min = results['val_matrix']

        val_min[abs(val_min) < remove_link_threshold] = 0  # set values below threshold to zero
        graph[abs(val_min) < remove_link_threshold] = ""  # set values below threshold to zero

        # plot predicted PAG
        tp.plot_graph(
            val_matrix=val_min,
            link_matrix=graph,
            var_names=var_names,
            link_colorbar_label='cross-MCI',
            node_colorbar_label='auto-MCI',
            figsize=(10, 6),
        )
        plt.show()

        # save results
        save_results(val_min, graph, var_names, 'simulated')
        return val_min, graph, var_names