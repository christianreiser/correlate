from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from data_cleaning_and_imputation import missing_value_check, data_cleaning_and_imputation, \
    drop_attributes_with_missing_values, drop_days_before__then_drop_col, drop_days_with_missing_values


def main():
    # label of interest
    target_label = 'mood'
    show_plots = False
    load_precomputed_coefficients_and_p_val = True

    # load data
    df = pd.read_csv('/home/chrei/code/quantifiedSelfData/daily_summaries.csv', index_col=0)

    # autocorrelation(df, target_label)

    # data cleaning and imputation
    df = data_cleaning_and_imputation(df, target_label)

    # correlation and p value
    results = corr_coefficients_and_p_values(df, target_label, show_plots,
                                             load_precomputed_coefficients_and_p_val)

    # histograms(df, save_path='/home/chrei/PycharmProjects/correlate/plots/distributions/')

    # std normalization preprocessing
    # df = (df - df.mean()) / df.std()  # built in normalization not used
    # histograms(df, save_path='/home/chrei/PycharmProjects/correlate/plots/distributions_after_normalization/')

    # single linear regression
    # single_linear_regression(df, target_label, results) # todo: continue implementation

    # multiple linear regression on different datasets
    df_longest = drop_attributes_with_missing_values(df)
    multiple_regression(df_longest, target_label, results, dataset_name='longest')

    df_2019_09_08 = drop_days_before__then_drop_col(df, last_day_to_drop='2019-09-08')
    multiple_regression(df_2019_09_08, target_label, results, dataset_name='after2019_09_08')

    df_widest = drop_days_with_missing_values(df)
    multiple_regression(df_widest, target_label, results, dataset_name='widest')

    print('done')


def drop_days_where_mood_was_tracked_irregularly(df):
    date_list = pd.date_range(start=datetime.strptime('2019-02-11', '%Y-%m-%d'), end='2019-08-29').tolist()
    date_list = [day.strftime('%Y-%m-%d') for day in date_list]
    for date in date_list:
        try:
            df = df.drop(date, axis=0)
        except:
            pass

    date_list = pd.date_range(start=datetime.strptime('2021-06-15', '%Y-%m-%d'), end='2021-07-26').tolist()
    date_list = [day.strftime('%Y-%m-%d') for day in date_list]
    for date in date_list:
        try:
            df = df.drop(date, axis=0)
        except:
            pass
    return df


def autocorrelation(df, target_label):
    target_df = df[target_label]

    target_df = drop_days_where_mood_was_tracked_irregularly(target_df)

    # Autocorrelation 654 lags
    plot_acf(target_df, title=str(target_label) + ' autocorrelation with 95% confidence interval', lags=654, alpha=.05)
    plt.savefig('/home/chrei/PycharmProjects/correlate/plots/autocorrelation_650lags_' + str(target_label))

    # Autocorrelation 250 lags
    plot_acf(target_df, title=str(target_label) + ' autocorrelation with 95% confidence interval', lags=250, alpha=.05)
    plt.savefig('/home/chrei/PycharmProjects/correlate/plots/autocorrelation_250lags_' + str(target_label))

    # Autocorrelation 50 lags
    plot_acf(target_df, title=str(target_label) + ' autocorrelation with 95% confidence interval', lags=50, alpha=.05)
    plt.savefig('/home/chrei/PycharmProjects/correlate/plots/autocorrelation_050lags_' + str(target_label))

    # partial Autocorrelation 326 lags
    plot_pacf(target_df, lags=326, alpha=.05,
              title=str(target_label) + ' partial autocorrelation with 95% confidence interval')
    plt.savefig('/home/chrei/PycharmProjects/correlate/plots/partial_autocorrelation_326lags_' + str(target_label))

    # partial Autocorrelation 25 lags
    plot_pacf(target_df, lags=25, alpha=.05,
              title=str(target_label) + ' partial autocorrelation with 95% confidence interval')
    plt.savefig('/home/chrei/PycharmProjects/correlate/plots/partial_autocorrelation_025lags_' + str(target_label))


def histograms(df, save_path):
    # plot distributions
    for attribute in df.columns:
        print(attribute)
        # attribute = 'VO2Max'
        n, bins, patches = plt.hist(df[attribute], 50, density=True, facecolor='g', alpha=0.75)
        # plt.xlabel('#')
        # plt.ylabel('Probability')
        plt.title('Histogram of ' + str(attribute))
        # plt.xlim(40, 160)
        # plt.ylim(0, 0.03)
        plt.grid(True)
        # plt.show()
        plt.savefig(save_path + str(attribute), dpi=None,
                    facecolor='w',
                    edgecolor='w',
                    orientation='portrait', format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    metadata=None)
        plt.close('all')


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
    df = drop_days_where_mood_was_tracked_irregularly(df)
    missing_value_check(df)

    y = df[target_label]
    predictions_results = y
    X = df.drop([target_label], axis=1)

    # time series split
    tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
    i = 0
    cross_validation_loss_list = []
    for train_index, test_index in tscv.split(X):
        i += 1
        # print("TRAIN:", train_index, "\nTEST:", test_index)
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        regression = linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False)  # already normalized
        regression.fit(X_train, y_train)
        # x_labels = X.columns
        regression_coefficient_df = pd.DataFrame(index=X.columns, columns=['reg_coeff'])
        regression_coefficient_df['reg_coeff'] = regression.coef_
        print('intercept:', regression.intercept_, dataset_name)
        results['reg_coeff_' + str(dataset_name) + 'k=' + str(i)] = regression_coefficient_df
        results.to_csv('/home/chrei/code/quantifiedSelfData/results.csv')  # save to file
        predictions = regression.predict(X_test)
        predictions = pd.DataFrame(list(zip(y_test.index, predictions)), columns =['date', str(dataset_name) + ' k=' + str(i)])
        predictions = predictions.set_index('date')
        predictions_results = predictions_results.join(predictions)
        l1_loss = abs(y_test - predictions)
        mean_l1_loss_for_one_fold = l1_loss.mean(axis=0)
        print('L1 loss ' + str(dataset_name) + 'k=' + str(i), ': ', mean_l1_loss_for_one_fold)
        cross_validation_loss_list.append(mean_l1_loss_for_one_fold)
    cross_validation_loss = np.mean(cross_validation_loss_list)
    print('cross_validation_loss: ', cross_validation_loss)


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
