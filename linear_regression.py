import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
import pandas as pd

from config import target_label, out_of_bound_correction_on, regularization_strength, l1_ratio, ensemble_weights
from helper import histograms, plot_prediction_w_ci_interval, drop_days_where_mood_was_tracked_irregularly, \
    out_of_bound_correction


def multiple_linear_regression_ensemble(df, df_longest, df_2019_09_08, df_widest, results,
                                        target_lower_bound_normalized, target_upper_bound_normalized, target_mean,
                                        target_std):
    # multiple linear regression on different datasets
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

            regression = ElasticNet(alpha=regularization_strength, l1_ratio=l1_ratio, fit_intercept=True,
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
