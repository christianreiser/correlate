import numpy as np
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI

from causal_discovery.LPCMCI.lpcmci import LPCMCI
from config import verbosity_thesis, causal_discovery_on, tau_max, pc_alpha, private_folder_path, LPCMCI_or_PCMCI, \
    remove_link_threshold, verbosity

"""
plain causal discovery
"""


# function that saves val_min, graph, and var_names to a file
def save_results(val_min, graph, var_names, name_extension):
    np.save(str(private_folder_path) + 'val_min_' + str(name_extension), val_min)
    np.save(str(private_folder_path) + 'graph_' + str(name_extension), graph)
    np.save(str(private_folder_path) + 'var_names_' + str(name_extension), var_names)


def if_intervened_replace_with_nan(ts, was_intervened):
    # iterate over all rows and columns of ts
    for i in range(len(ts)):
        for j in range(len(ts.columns)):
            if was_intervened.iloc[i, j]:
                ts.iloc[i, j] = np.NaN
    return ts


def observational_causal_discovery(df, was_intervened, external_independencies):
    """
    1. get observational ts
    2. ini graph with previous pag_edgemarks and pag_effect_sizes
    3. reduce pag_edgemarks with observatonal data and update pag_effect_sizes
    return: pag_edgemarks, pag_effect_sizes
    """
    if causal_discovery_on:

        """ below code is only needed for real world data"""
        """get non_zero_indices"""
        """
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
        """
        # handle interventions: in df set value to NaN if it was intervened
        # during CI tests nans are then excluded
        df = if_intervened_replace_with_nan(df, was_intervened)

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
                external_independencies = external_independencies,
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
                verbosity=verbosity)

            results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
            q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh',
                                                   exclude_contemporaneous=False)

            graph = results['graph']
            val_min = results['val_matrix']

        # remove links if are below threshold
        graph[abs(val_min) < remove_link_threshold] = ""

        # # plot predicted PAG
        # tp.plot_graph(
        #     val_matrix=val_min,
        #     link_matrix=graph,
        #     var_names=var_names,
        #     link_colorbar_label='cross-MCI',
        #     node_colorbar_label='auto-MCI',
        #     figsize=(10, 6),
        # )
        # plt.show()

        # save results
        save_results(val_min, graph, var_names, 'simulated')
        return val_min, graph


# # load ts dataframe from file
# import os
#
# filename = os.path.abspath("./tmp_test.dat")
# ts = pd.read_csv(filename, index_col=0)
#
# # get last row of ts and append to ts
# ts = ts.append(ts.iloc[-1])
#
# ## load was_intervened dataframe from file
# filename = os.path.abspath("./tmp_was_intervened.dat")
# was_intervened = pd.read_csv(filename, index_col=0)
#
# pag_effect_sizes, pag_edgemarks = observational_causal_discovery(
#     pag_edgemarks=None,
#     pag_effect_sizes=None,
#     df=ts,
#     was_intervened=was_intervened)
#
# print()
