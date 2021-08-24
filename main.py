from datetime import datetime

import numpy as np
import pandas as pd
import seaborn
import seaborn as sns
import statsmodels.formula.api as sm
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from data_cleaning_and_imputation import drop_attributes_with_missing_values, drop_days_before__then_drop_col, \
    drop_days_with_missing_values, data_cleaning_and_imputation
from helper import histograms


def main():
    # label of interest
    target_label = 'mood'
    target_lower_bound = 1.0
    target_upper_bound = 9.0
    show_plots = False  # corr matrix
    plot_distributions = False
    load_precomputed_coefficients_and_p_val = True
    add_yesterdays_target_feature = False
    add_ereyesterdays_target_feature = True
    add_all_yesterdays_features = True
    out_of_bound_correction_on = False
    ensemble_weights = [0, 0.4, 0.6]  # [longest, compromise, widest]
    regularization_strength = 0.12
    l1_ratio = 1

    checks(ensemble_weights)

    # load data
    df = pd.read_csv('/home/chrei/code/quantifiedSelfData/daily_summaries_all.csv', index_col=0)

    # autocorrelation(df, target_label)

    # data cleaning and imputation
    df = data_cleaning_and_imputation(df, target_label, add_all_yesterdays_features, add_yesterdays_target_feature,
                                      add_ereyesterdays_target_feature)

    # correlation and p value
    results = corr_coefficients_and_p_values(df, target_label, show_plots,
                                             load_precomputed_coefficients_and_p_val)

    if plot_distributions:
        histograms(df, save_path='/home/chrei/PycharmProjects/correlate/plots/distributions/')

    # std normalization preprocessing
    df_mean = df.mean()
    df_std = df.std()
    target_mean = df_mean[target_label]
    target_std = df_std[target_label]
    df = (df - df_mean) / df_std  # built in normalization not used
    print('target_mean:', target_mean)
    print('target_std:', target_std)
    target_upper_bound_normalized = (target_upper_bound - df_mean[target_label]) / df_std[target_label]
    target_lower_bound_normalized = (target_lower_bound - df_mean[target_label]) / df_std[target_label]

    if plot_distributions:
        histograms(df, save_path='/home/chrei/PycharmProjects/correlate/plots/distributions_after_normalization/')

    # # multiple linear regression on different datasets
    df_longest = drop_attributes_with_missing_values(df)
    df_2019_09_08 = drop_days_before__then_drop_col(df, last_day_to_drop='2019-09-08')
    df_widest = drop_days_with_missing_values(df, add_all_yesterdays_features)
    predictions_results = df[target_label].to_frame()

    predictions_results = multiple_regression(df_longest, target_label, results, out_of_bound_correction_on,
                                              dataset_name='longest',
                                              predictions_results=predictions_results,
                                              target_lower_bound=target_lower_bound_normalized,
                                              target_upper_bound=target_upper_bound_normalized,
                                              regularization_strength=regularization_strength, l1_ratio=l1_ratio)

    predictions_results = multiple_regression(df_2019_09_08, target_label, results, out_of_bound_correction_on,
                                              dataset_name='after2019_09_08',
                                              predictions_results=predictions_results,
                                              target_lower_bound=target_lower_bound_normalized,
                                              target_upper_bound=target_upper_bound_normalized,
                                              regularization_strength=0.07, l1_ratio=0.9)  # 0.07 0.6530

    predictions_results = multiple_regression(df_widest, target_label, results, out_of_bound_correction_on,
                                              dataset_name='widest',
                                              predictions_results=predictions_results,
                                              target_lower_bound=target_lower_bound_normalized,
                                              target_upper_bound=target_upper_bound_normalized,
                                              regularization_strength=0.12, l1_ratio=l1_ratio)
    # predictions_results = pd.read_csv('/home/chrei/code/quantifiedSelfData/predictions_results.csv', index_col=0)

    predictions_results['ensemble_prediction'] = ensemble_weights[0] * predictions_results['longest k=5'] + \
                                                 ensemble_weights[1] * predictions_results[
                                                     'after2019_09_08 k=5'] + ensemble_weights[2] * predictions_results[
                                                     'widest k=5']

    # diff
    predictions_results['ensemble_diff'] = predictions_results['mood'] - predictions_results[
        'ensemble_prediction']
    histograms(predictions_results['ensemble_diff'].to_frame(),
               save_path='/home/chrei/PycharmProjects/correlate/plots/prediction_diff/')

    # l1
    predictions_results['ensemble_residuals'] = abs(predictions_results['ensemble_diff'])
    histograms(predictions_results['ensemble_residuals'].to_frame(),
               save_path='/home/chrei/PycharmProjects/correlate/plots/prediction_diff/')
    ensemble_average_residual = predictions_results['ensemble_residuals'].mean()
    print('ensemble_average_residual: ', ensemble_average_residual)
    predictions_results['CI_low'] = predictions_results['ensemble_prediction'] - ensemble_average_residual
    predictions_results['CI_high'] = predictions_results['ensemble_prediction'] + ensemble_average_residual
    ci = np.percentile(predictions_results['ensemble_residuals'].dropna(), 95)
    print('prediction 95% confidence interval: ', ci)
    plot_prediction_w_ci_interval(predictions_results, ci, target_mean, target_std)

    # l2
    predictions_results['ensemble_loss'] = predictions_results['ensemble_diff'] ** 2
    ensemble_average_loss = predictions_results['ensemble_loss'].mean()
    print('ensemble_average_loss: ', ensemble_average_loss)

    # save
    predictions_results.to_csv('/home/chrei/code/quantifiedSelfData/predictions_results.csv')  # save to file

    # PCA
    # pca_function(df_longest)
    print('done')


def plot_prediction_w_ci_interval(df, ci, target_mean, target_std):
    df = df.copy().dropna()
    df.reset_index(level=0, inplace=True)
    df['prediction_not_normalized'] = df['ensemble_prediction'].multiply(target_std).add(target_mean)
    df['mood_not_normalized'] = df['mood'] * target_std + target_mean
    sns.set_theme(style="darkgrid")
    sns.set(rc={'figure.figsize': (11.7, 8.27)})

    sns.pointplot(x="prediction_not_normalized", y="Date", data=df, join=False, color='r', label="prediction_not_normalized")
    sns.pointplot(x="mood_not_normalized", y="Date", data=df, join=False, color='g', label="mood_not_normalized")
    plt.errorbar(df['prediction_not_normalized'], df['Date'],
                 xerr=np.ones(len(df.loc[:, 'Date'])) * ci * target_std)
    # plt.legend(labels=['legendEntry1', 'legendEntry2'])

    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='#bb3f3f', label='prediction')
    black_patch = mpatches.Patch(color='#009152', label='ground truth')
    plt.legend(handles=[red_patch, black_patch], loc="upper left")

    plt.tight_layout()
    plt.xlim(0.9, 9.1)
    plt.savefig('/home/chrei/PycharmProjects/correlate/plots/predictions', dpi=200)
    plt.close('all')


def checks(ensemble_weights):
    if not sum(ensemble_weights) == 1.0:
        raise ValueError('sum(ensemble_weights) != 1.0')


def pca_function(df):
    n_components = len(df.columns)
    pca = PCA(n_components=n_components)
    pca.fit(df)
    PCA(n_components=n_components)
    print(pca.explained_variance_ratio_)

    plt.plot(pca.explained_variance_ratio_, alpha=0.75)
    plt.xlabel('component')
    plt.ylabel('explained variance ratio')
    plt.title('PCA explained variance ratio')
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    plt.grid(True)
    # plt.show()

    plt.savefig('/home/chrei/PycharmProjects/correlate/plots/pca_explained_variance_ratio', dpi=None,
                facecolor='w',
                edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                metadata=None)
    plt.close('all')

    # visualization
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlabel('Principal Component 1', fontsize=15)
    # ax.set_ylabel('Principal Component 2', fontsize=15)
    # ax.set_title('2 component PCA', fontsize=20)
    # targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    # colors = ['r', 'g', 'b']
    # for target, color in zip(targets, colors):
    #     indicesToKeep = finalDf['target'] == target
    #     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
    #                , finalDf.loc[indicesToKeep, 'principal component 2']
    #                , c=color
    #                , s=50)
    # ax.legend(targets)
    # ax.grid()


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


def multiple_regression(df, target_label, results, out_of_bound_correction_on, dataset_name, predictions_results,
                        target_upper_bound,
                        target_lower_bound, regularization_strength, l1_ratio):
    df = drop_days_where_mood_was_tracked_irregularly(df)
    # missing_value_check(df)

    y = df[target_label]
    X = df.drop([target_label], axis=1)

    # time series split
    tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
    i = 0
    cross_validation_loss_list = []
    for train_index, test_index in tscv.split(X):
        i += 1
        if i == 5:  # CV in time series neglects too much training data, thus use a simple train test split
            # print("TRAIN:", train_index, "\nTEST:", test_index)
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            regression = linear_model.ElasticNet(alpha=regularization_strength, l1_ratio=l1_ratio, fit_intercept=True,
                                                 normalize=False)  # already normalized
            regression.fit(X_train, y_train)
            # x_labels = X.columns
            regression_coefficient_df = pd.DataFrame(index=X.columns, columns=['reg_coeff'])
            regression_coefficient_df['reg_coeff'] = regression.coef_
            # print('intercept:', regression.intercept_, dataset_name)
            results['reg_coeff_' + str(dataset_name) + 'k=' + str(i)] = regression_coefficient_df
            results.to_csv('/home/chrei/code/quantifiedSelfData/results.csv')  # save to file
            predictions = regression.predict(X_test)
            predictions = pd.DataFrame(list(zip(y_test.index, predictions)),
                                       columns=['date', str(dataset_name) + ' k=' + str(i)])
            predictions = predictions.set_index('date')
            predictions = out_of_bound_correction(predictions, target_upper_bound, target_lower_bound,
                                                  on=out_of_bound_correction_on)

            predictions_results = predictions_results.join(predictions)
            l2_loss = (y_test - predictions[str(dataset_name) + ' k=' + str(i)]) ** 2
            mean_l2_loss_for_one_fold = l2_loss.mean(axis=0)
            # print('L2 loss ' + str(dataset_name) + 'k=' + str(i), ': ', mean_l2_loss_for_one_fold)
            cross_validation_loss_list.append(mean_l2_loss_for_one_fold)
    cross_validation_loss = np.mean(cross_validation_loss_list)
    print('cross_validation_loss: ', cross_validation_loss)
    return predictions_results


def out_of_bound_correction(predictions, target_upper_bound, target_lower_bound, on):
    if on:
        # correct if prediction is out of bounds
        for day, i in predictions.iterrows():
            prediction = predictions[predictions.columns[0]][day]
            if prediction > target_upper_bound:
                print('out_of_bound_correction: predictions[i]: ', prediction, 'target_upper_bound:',
                      target_upper_bound)
                correction = target_upper_bound
                predictions[predictions.columns[0]][day] = correction

            elif prediction < target_lower_bound:
                print('out_of_bound_correction: predictions[i]: ', prediction, 'target_lower_bound:',
                      target_lower_bound)
                correction = target_lower_bound
                predictions[predictions.columns[0]][day] = correction
    return predictions


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
        np.fill_diagonal(corr_matrix.values, np.nan)
        # test = corr_matrix.shape[0]
        # corr_matrix[np.tril_indices(4)] = np.nan
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

    # correlation p value scatter plot
    results['corr_coeff_abs'] = results['corrCoeff'].abs()
    seaborn.scatterplot(data=results, x="corr_coeff_abs", y="pvals_corrected")
    plt.title('Corr pVal scatter plot')
    # plt.yscale('log')
    plt.savefig('/home/chrei/PycharmProjects/correlate/plots/corr_pVal_scatter')
    plt.close('all')
    return results


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


main()