import numpy as np
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI

from causal_discovery.helper import reduce_tau_max
from causal_discovery.preprocessing import remove_nan_seq_from_top_and_bot, non_contemporary_tie_series_generation
from config import verbosity, alpha_level, causal_discovery_on


def causal_discovery(df, tau_max):
    if causal_discovery_on:
        # df = pd.read_csv('./data/daily_summaries_all (copy).csv', sep=",")
        # df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

        # select columns
        df = df[['Date', 'SleepEfficiency', 'Mood']]

        df = remove_nan_seq_from_top_and_bot(df)
        df = non_contemporary_tie_series_generation(df)
        df = df.drop(['Date'], axis=1)  # drop date col

        # # standardize data
        df -= df.mean(axis=0)
        df /= df.std(axis=0)

        var_names = df.columns
        dataframe = pp.DataFrame(df.values, datatime=np.arange(len(df)),
                                 var_names=var_names)

        # tp.plot_timeseries(dataframe)
        # plt.show()

        parcorr = ParCorr(significance='analytic')
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=parcorr,
            verbosity=verbosity)

        correlations = pcmci.get_lagged_dependencies(tau_max=tau_max, val_only=True)['val_matrix']
        lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations,
                                           setup_args={'var_names': var_names, 'x_base': 5, 'y_base': .5})
        plt.show()

        tau_max = reduce_tau_max(correlations)

        pcmci.verbosity = verbosity
        results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None)

        q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=tau_max, fdr_method='fdr_bh')
        pcmci.print_significant_links(
            p_matrix=results['p_matrix'],
            q_matrix=q_matrix,
            val_matrix=results['val_matrix'],
            alpha_level=alpha_level)

        link_matrix = pcmci.return_significant_links(pq_matrix=q_matrix,
                                                     val_matrix=results['val_matrix'], alpha_level=alpha_level)[
            'link_matrix']
        tp.plot_graph(
            val_matrix=results['val_matrix'],
            link_matrix=link_matrix,
            var_names=var_names,
            link_colorbar_label='cross-MCI',
            node_colorbar_label='auto-MCI',
            figsize=(10, 6),
        )
        plt.show()

        # Plot time series graph
        tp.plot_time_series_graph(
            figsize=(12, 8),
            val_matrix=results['val_matrix'],
            link_matrix=link_matrix,
            var_names=var_names,
            link_colorbar_label='MCI',
        )
        plt.show()
