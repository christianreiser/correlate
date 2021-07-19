import pandas as pd
import statsmodels.formula.api as sm
from matplotlib import pyplot as plt
from sklearn import linear_model
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from data_cleaning_and_imputation import data_cleaning_and_imputation, \
    drop_attributes_with_missing_values, drop_days_with_missing_values, drop_days_before__then_drop_col, \
    missing_value_check


def main():
    # label of interest
    target_label = 'mood'
    show_plots = False
    load_precomputed_coefficients_and_p_val = True

    # load data
    df = pd.read_csv('/home/chrei/code/quantifiedSelfData/daily_summaries.csv', index_col=0)

    # data cleaning and imputation
    df = data_cleaning_and_imputation(df, target_label)

    # correlation and p value
    results = corr_coefficients_and_p_values(df, target_label, show_plots,
                                             load_precomputed_coefficients_and_p_val)

    # single linear regression
    # single_linear_regression(df, target_label, results) # todo: continue implementation

    # multiple linear regression
    df_longest = drop_attributes_with_missing_values(df)
    multiple_regression(df_longest, target_label, results, dataset_name='longest')

    df_2019_09_08 = drop_days_before__then_drop_col(df, last_day_to_drop='2019-09-08')
    multiple_regression(df_2019_09_08, target_label, results, dataset_name='after2019_09_08')

    df_widest = drop_days_with_missing_values(df)
    multiple_regression(df_widest, target_label, results, dataset_name='widest')

    print('done')


def single_linear_regression(df, target_label, results):
    # todo: continue implementation
    y = df[target_label]
    x = df.drop([target_label], axis=1)
    regression_coefficient_df = pd.DataFrame(index=x.columns, columns=['single_reg_coeff'])

    for column in x.columns:
        regression = linear_model.LinearRegression()
        xCol = x[column]
        regression.fit(x[column], y)  # todo there is a bug
        coef = regression.coef_

    regression_coefficient_df['single_reg_coeff'][column] = regression.coef_
    results['single_reg_coeff_'] = regression_coefficient_df
    results.to_csv('/home/chrei/code/quantifiedSelfData/results.csv')  # save to file


def multiple_regression(df, target_label, results, dataset_name):
    missing_value_check(df)
    y = df[target_label]
    x = df.drop([target_label], axis=1)
    regression = linear_model.LinearRegression()
    regression.fit(x, y)
    # x_labels = X.columns
    regression_coefficient_df = pd.DataFrame(index=x.columns, columns=['reg_coeff'])
    regression_coefficient_df['reg_coeff'] = regression.coef_
    results['reg_coeff_' + str(dataset_name)] = regression_coefficient_df
    results.to_csv('/home/chrei/code/quantifiedSelfData/results.csv')  # save to file


def p_values(corr_matrix, df, target_label):
    # p-values
    p_val_matrix = corr_matrix.copy()
    print('computing p values. TODO: Is there a faster way?')
    for i in tqdm(range(df.shape[1])):  # rows are the number of rows in the matrix.
        for j in range(df.shape[1]):
            y = df.columns[i]
            x = df.columns[j]
            df_ols = sm.ols(formula='Q("{}") ~ Q("{}")'.format(y, x), data=df).fit()
            p_val_matrix.iloc[i, j] = df_ols.pvalues[1]
    target_p_values = p_val_matrix[target_label]  # get target label from matrix
    target_p_values = target_p_values.drop([target_label])  # drop self correlation
    return target_p_values


def visualize_corr_matrix(corr_matrix, df, show):
    if show:
        # plot
        f = plt.figure(figsize=(19, 15))
        plt.matshow(corr_matrix, fignum=f.number)
        plt.xticks(range(df.shape[1]), df.columns, fontsize=7, rotation=90)
        plt.yticks(range(df.shape[1]), df.columns, fontsize=7)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=7)
        plt.title('Correlation Matrix', fontsize=12)
        plt.show()


def visualize_corr_and_p_values(corr_coeff_and_p_val, show):
    if show:
        # plot
        # corr_coeff_and_p_val = corr_coeff_and_p_val.set_index(['Unnamed: 0'])
        # corr_coeff_and_p_val = corr_coeff_and_p_val.T
        # f = plt.figure(figsize=(20, 20))
        # plt.matshow(corr_coeff_and_p_val.T, fignum=f.number)
        # plt.yticks(range(0, corr_coeff_and_p_val.shape[1]), corr_coeff_and_p_val.columns, fontsize=7)
        # plt.xticks(range(0, 2), ['corrCoeff', 'pVal'], fontsize=7,rotation=90)
        # cb = plt.colorbar()
        # cb.ax.tick_params(labelsize=7)
        # plt.title('Correlation Matrix', fontsize=12)
        # plt.show()
        headers = corr_coeff_and_p_val.index
        corr = corr_coeff_and_p_val['corrCoeff']
        pval = corr_coeff_and_p_val['pVal']
        plt.plot(corr_coeff_and_p_val.index, corr_coeff_and_p_val['corrCoeff'], 'g^', corr_coeff_and_p_val.index,
                 corr_coeff_and_p_val['pVal'], 'bs')
        plt.xticks(rotation=90)

        plt.show()


def corr_coefficients_and_p_values(df, target_label, show_plots, load_precomputed_values):
    # load precomputed values
    if load_precomputed_values:
        results = pd.read_csv('/home/chrei/code/quantifiedSelfData/results.csv', index_col=0)

    # compute correlations and p values
    else:
        # correlate
        corr_matrix = pd.DataFrame.corr(df, method='pearson', min_periods=5)
        visualize_corr_matrix(corr_matrix, df, show_plots)

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

        results.to_csv('/home/chrei/code/quantifiedSelfData/results.csv')

    # visualize
    visualize_corr_and_p_values(results, show_plots)

    return results


main()
