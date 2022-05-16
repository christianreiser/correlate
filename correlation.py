from multiprocessing.pool import ThreadPool as Pool

import numpy as np
import pandas as pd
import seaborn
import statsmodels.formula.api as sm
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from config import show_plots, load_precomputed_coefficients_and_p_val, private_folder_path


def corr_coefficients_and_p_values(df, target_label):
    # load precomputed values
    if load_precomputed_coefficients_and_p_val:
        results = pd.read_csv(str(private_folder_path) + 'results.csv', index_col=0)

    # compute correlations and p values
    else:
        # correlate
        corr_matrix = pd.DataFrame.corr(df, method='pearson', min_periods=5)
        np.fill_diagonal(corr_matrix.values, np.nan)
        # test = corr_matrix.shape[0]
        # corr_matrix[np.tril_indices(4)] = np.nan
        visualize_corr_matrix(corr_matrix, df)

        # get target values
        target_correlations = corr_matrix[target_label]  # get target label from matrix
        target_correlations = target_correlations.drop([target_label])  # drop self correlation

        # compute p values
        target_p_values = p_values(corr_matrix, df, target_label)

        # combine to single df
        results = pd.DataFrame(index=target_p_values.index, columns=['corrCoeff', 'pVal'])
        results['pVal'] = target_p_values
        results['corrCoeff'] = target_correlations

        # sort by p Val
        results = results.sort_values(kind="quicksort", by='pVal')

        # Benjamini–Hochberg procedure
        reject_0_hypothesis, pvals_corrected, alphacSidak, alphacBonf = multipletests(results['pVal'], alpha=0.05,
                                                                                      method='fdr_bh', is_sorted=False,
                                                                                      returnsorted=False)
        results['pvals_corrected'] = pvals_corrected
        results['reject_0_hypothesis'] = reject_0_hypothesis

        results.to_csv(str(private_folder_path) + 'results.csv')

    # visualize
    visualize_corr_and_p_values(results)

    # correlation p value scatter plot
    results['corr_coeff_abs'] = results['corrCoeff'].abs()
    seaborn.scatterplot(data=results, x="corr_coeff_abs", y="pvals_corrected")
    plt.title('Corr pVal scatter plot')
    # plt.yscale('log')
    plt.savefig('/home/chrei/PycharmProjects/correlate/plots/corr_pVal_scatter')
    plt.close('all')
    return results


def worker1(i, df, p_val_matrix):
    for j in range(df.shape[1]):
        y = df.columns[i]
        x = df.columns[j]
        df_ols = sm.ols(formula='Q("{}") ~ Q("{}")'.format(y, x), data=df).fit()
        p_val_matrix.iloc[i, j] = df_ols.pvalues[1]


def p_values(corr_matrix, df, target_label):
    # p-values
    p_val_matrix = corr_matrix.copy()
    print('computing p values. TODO: Is there a faster way?')

    pool_size = 12  # your "parallelness"
    pool = Pool(pool_size)
    i = -1
    for column in tqdm(df.columns):  # rows are the number of rows in the matrix.
        i += 1
        pool.apply_async(worker1(i, df, p_val_matrix), (column,))

    pool.close()
    pool.join()

    target_p_values = p_val_matrix[target_label]  # get target label from matrix
    target_p_values = target_p_values.drop([target_label])  # drop self correlation
    return target_p_values


def visualize_corr_matrix(corr_matrix, df):
    if show_plots:
        # plot
        f = plt.figure(figsize=(19, 15))
        plt.matshow(corr_matrix, fignum=f.number)
        plt.xticks(range(df.shape[1]), df.columns, fontsize=7, rotation=90)
        plt.yticks(range(df.shape[1]), df.columns, fontsize=7)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=7)
        plt.title('Correlation Matrix', fontsize=12)
        plt.show()


def visualize_corr_and_p_values(corr_coeff_and_p_val):
    """
    there is some error but only when running in debug mode?
    """
    if show_plots:
        i = corr_coeff_and_p_val.index
        c = corr_coeff_and_p_val['corrCoeff']
        p = corr_coeff_and_p_val['pVal']
        plt.plot(i, c, 'g^', i, p, 'bs')
        plt.xticks(rotation=90)

        plt.show()
