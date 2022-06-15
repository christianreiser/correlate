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

"""
plain causal discovery
"""


# function that saves val_min, graph, and var_names to a file
def save_results(val_min, graph, var_names, name_extension):
    np.save(str(private_folder_path) + 'val_min_'+str(name_extension), val_min)
    np.save(str(private_folder_path) + 'graph_'+str(name_extension), graph)
    np.save(str(private_folder_path) + 'var_names_'+str(name_extension), var_names)


# def observational_causal_discovery(pag_edgemarks, pag_effect_sizes, df, was_intervened):
def observational_causal_discovery(df):

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

# # load ts dataframe from file
# import os
# filename = os.path.abspath("./LPCMCI/tmp_test.dat")
# fileobj = open(filename, mode='rb')
# ts = np.fromfile(fileobj, dtype=np.float32)
# fileobj.close()
#
# ## load was_intervened dataframe from file
# import os
# filename = os.path.abspath("./tmp_was_intervened.dat")
# was_intervened = pd.read_csv(filename, index_col=0)
# print()
#
#
# pag_effect_sizes, pag_edgemarks, var_names = observational_causal_discovery(
#     pag_edgemarks='fully connected',
#     pag_effect_sizes=None,
#     df=ts,
#     was_intervened  =was_intervened)