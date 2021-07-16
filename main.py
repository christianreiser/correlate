import pandas as pd
import statsmodels.formula.api as sm
from matplotlib import pyplot as plt
from sklearn import linear_model
from tqdm import tqdm

from data_cleaning_and_imputation import data_cleaning_and_imputation


def main():
    # label of interest
    target_label = 'mood'
    show_correlation_matrix = False
    load_precomputed_values = True

    # load data
    df = pd.read_csv('/home/chrei/code/quantifiedSelfData/daily_summaries.csv', index_col=0)

    df = data_cleaning_and_imputation(df, target_label)

    corr_coeff_and_p_val = corr_coeffs_and_p_values(df, target_label, show_correlation_matrix, load_precomputed_values)

    multiple_regression(df, target_label)
    print('break')


def multiple_regression(df, target_label):
    y = df[target_label]
    X = df.drop([target_label], axis=1)
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    # X_lables = X.columns
    regr_coef_df = pd.DataFrame(index=X.columns, columns=['regrCoef'])
    regr_coef_df['regrCoef'] = regr.coef_
    print('break')


def pValues(corr_matrix, df, target_label):
    # p-values
    p_val_matrix = corr_matrix.copy()
    print('computing p values. TODO: Is there a faster way?')
    for i in tqdm(range(df.shape[1])):  # rows are the number of rows in the matrix.
        for j in range(df.shape[1]):
            y = df.columns[i]
            x = df.columns[j]
            df_ols = sm.ols(formula='Q("{}") ~ Q("{}")'.format(y, x), data=df).fit()
            p_val_matrix.iloc[i, j] = df_ols.pvalues[1]
    # p_val_matrix = pd.read_csv('/home/chrei/code/quantifiedSelfData/target_p_values.csv')
    target_p_values = p_val_matrix[target_label]  # get target label from metrix
    target_p_values = target_p_values.drop([target_label])  # drop self correlation
    return target_p_values


def corr_coeffs_and_p_values(df, target_label, show_correlation_matrix, load_precomputed_values):
    if load_precomputed_values:
        corr_coeff_and_p_val = pd.read_csv('/home/chrei/code/quantifiedSelfData/corrCoeff_and_pVal.csv')

    else:
        # correlate
        corr_matrix = pd.DataFrame.corr(df, method='pearson', min_periods=5)
        target_correlations = corr_matrix[target_label]  # get target label from matrix
        target_correlations = target_correlations.drop([target_label])  # drop self correlation

        # compute p values
        target_p_values = pValues(corr_matrix, df, target_label)

        # combine to single df
        corr_coeff_and_p_val = pd.DataFrame(index=target_p_values.index, columns=['corrCoeff', 'pVal'])
        corr_coeff_and_p_val['pVal'] = target_p_values
        corr_coeff_and_p_val['corrCoeff'] = target_correlations

        # sort by p Val
        corr_coeff_and_p_val = corr_coeff_and_p_val.sort_values(kind="quicksort", by='pVal')
        corr_coeff_and_p_val.to_csv('/home/chrei/code/quantifiedSelfData/corrCoeff_and_pVal.csv')

    if show_correlation_matrix:
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

    return corr_coeff_and_p_val


main()
