import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI

from causal_discovery.LPCMCI.lpcmci import LPCMCI
from causal_discovery.LPCMCI.observational_discovery import save_results
from config import verbosity, causal_discovery_on, tau_max, pc_alpha, private_folder_path, LPCMCI_or_PCMCI, \
    remove_link_threshold

"""
plain causal discovery
"""


def create_complete_graph(df):
    pass


def interv_discovery(df, was_intervened):

    # create complete graph
    create_complete_graph(df)

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
            max_p_non_ancestral=1, # todo 3
            n_preliminary_iterations=1, # todo 4
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

    # remove links if are below threshold
    graph[abs(val_min) < remove_link_threshold] = ""

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
    return val_min, graph


# load ts dataframe from file
import os

filename = os.path.abspath("./../LPCMCI/tmp_test.dat")
ts = pd.read_csv(filename, index_col=0)

# get last row of ts and append to ts
ts = ts.append(ts.iloc[-1])

## load was_intervened dataframe from file
filename = os.path.abspath("./../LPCMCI/tmp_was_intervened.dat")
was_intervened = pd.read_csv(filename, index_col=0)

pag_effect_sizes, pag_edgemarks = interv_discovery(
    df=ts,
    was_intervened=was_intervened)

print()
