import pandas as pd
import scipy.spatial.distance as dist
import scipy.stats as ss
from matplotlib import pyplot as plt
import statsmodels.formula.api as sm
from tqdm import tqdm


from data_cleaning_and_imputation import data_cleaning_and_imputation


def main():
    # label of interest
    target_label = 'mood'

    # load data
    df = pd.read_csv('/home/chrei/code/quantifiedSelfData/daily_summaries.csv', index_col=0)

    df = data_cleaning_and_imputation(df, target_label)

    target_correlations = correlation_matrix(df, target_label)

    print('break')


def correlation_matrix(df, target_label):

    # correlate
    corr_matrix = pd.DataFrame.corr(df, method='pearson', min_periods=5)

    # p-values
    pval_matrix = corr_matrix.copy()
    print('computing p values. TODO: Is there a faster way?')
    for i in tqdm(range(df.shape[1])):  # rows are the number of rows in the matrix.
        for j in range(df.shape[1]):
            y = df.columns[i]
            x = df.columns[j]
            df_ols = sm.ols(formula='Q("{}") ~ Q("{}")'.format(y, x), data=df).fit()
            pval_matrix.iloc[i, j] = df_ols.pvalues[1]

    target_p_values = pval_matrix[target_label]
    target_p_values = target_p_values.drop([target_label])

    target_correlations = corr_matrix[target_label]
    target_correlations = target_correlations.drop([target_label])
    target_correl_pval = pd.concat([target_correlations, target_correlations], axis=1)


    target_correlations = target_correlations.sort_values(kind="quicksort")

    # sort correlation matrix
    correlations = corr_matrix.unstack()
    correlations = correlations.sort_values(kind="quicksort")

    # plot
    f = plt.figure(figsize=(19, 15))
    plt.matshow(corr_matrix, fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=7, rotation=90)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=7)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=7)
    plt.title('Correlation Matrix', fontsize=12)
    plt.show()

    return target_correlations


main()
