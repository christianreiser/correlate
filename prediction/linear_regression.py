import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit

from config import target_label, ensemble_weights, multiple_linear_regression_ensemble_on, \
    regularization_strengths, l1_ratios, private_folder_path
from helper import histograms, plot_prediction_w_ci_interval, drop_days_where_mood_was_tracked_irregularly, \
    out_of_bound_correction, generate_sample_weights
from phone_io import write_csv_for_phone_visualization


def multiple_linear_regression_ensemble(df,
                                        df_not_normalized,
                                        df_longest,
                                        df_2019_09_08,
                                        df_widest,
                                        results,
                                        target_mean,
                                        target_std,
                                        target_scale_bounds_normalized,
                                        min_max):
    if multiple_linear_regression_ensemble_on:
        # multiple linear regression on different datasets
        prediction_results = df[target_label].to_frame()

        prediction_results = multiple_regression(df=df_longest,
                                                 results=results,
                                                 dataset_name='longest',
                                                 prediction_results=prediction_results,
                                                 regularization_strength=regularization_strengths[0],
                                                 l1_ratio=l1_ratios[0],
                                                 target_scale_bounds_normalized=target_scale_bounds_normalized)
        prediction_results = multiple_regression(df=df_2019_09_08,
                                                 results=results,
                                                 dataset_name='after2019_09_08',
                                                 prediction_results=prediction_results,
                                                 regularization_strength=regularization_strengths[1],
                                                 l1_ratio=l1_ratios[1],
                                                 target_scale_bounds_normalized=target_scale_bounds_normalized)

        prediction_results = multiple_regression(df=df_widest,
                                                 results=results,
                                                 dataset_name='widest',
                                                 prediction_results=prediction_results,
                                                 regularization_strength=regularization_strengths[2],
                                                 l1_ratio=l1_ratios[2],
                                                 target_scale_bounds_normalized=target_scale_bounds_normalized)

        prediction_results['ensemble_prediction'] = ensemble_weights[0] * prediction_results[
            'longest k=5'] + ensemble_weights[1] * prediction_results[
                                                        'after2019_09_08 k=5'
                                                    ] + ensemble_weights[2] * prediction_results['widest k=5']

        # diff
        prediction_results['ensemble_diff'] = prediction_results[target_label] - prediction_results[
            'ensemble_prediction']
        histograms(prediction_results['ensemble_diff'].to_frame(),
                   save_path='/home/chrei/PycharmProjects/correlate/plots/prediction_diff/')

        # l1
        prediction_results['ensemble_residuals'] = abs(prediction_results['ensemble_diff'])
        histograms(prediction_results['ensemble_residuals'].to_frame(),
                   save_path='/home/chrei/PycharmProjects/correlate/plots/prediction_diff/')
        ensemble_average_residual = prediction_results['ensemble_residuals'].mean()
        print('ensemble_average_residual: ', ensemble_average_residual)
        prediction_results['CI_low'] = prediction_results['ensemble_prediction'] - ensemble_average_residual
        prediction_results['CI_high'] = prediction_results['ensemble_prediction'] + ensemble_average_residual
        ci = np.percentile(prediction_results['ensemble_residuals'].dropna(), 95)
        ci68 = np.percentile(prediction_results['ensemble_residuals'].dropna(), 68)
        print('prediction 95% confidence interval: ', ci)
        plot_prediction_w_ci_interval(prediction_results, ci, target_mean, target_std)

        write_csv_for_phone_visualization(ci95=ci, ci68=ci68, target_mean=target_mean,
                                          prediction=prediction_results['widest k=5'],
                                          scale_bounds=min_max[target_label],
                                          feature_weights_normalized=results['reg_coeff_widestk=5'],
                                          feature_values_not_normalized=df_not_normalized,
                                          feature_values_normalized=df,
                                          target_std_dev=target_std,
                                          min_max=min_max)

        # l2
        prediction_results['ensemble_loss'] = prediction_results['ensemble_diff'] ** 2
        ensemble_average_loss = prediction_results['ensemble_loss'].mean()
        print('ensemble_average_loss: ', ensemble_average_loss)

        # save
        prediction_results.to_csv(str(private_folder_path) + 'prediction_results.csv')  # save to file


def multiple_regression(df, results, dataset_name, prediction_results, regularization_strength, l1_ratio,
                        target_scale_bounds_normalized):
    df = drop_days_where_mood_was_tracked_irregularly(df)

    


    # missing_value_check(df)

    y = df[target_label]
    X = df.drop([target_label], axis=1)

    # time series split for cv
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

            regression = ElasticNet(alpha=regularization_strength, l1_ratio=l1_ratio,
                                    fit_intercept=True)
            sample_weight = generate_sample_weights(y_train)
            regression.fit(X_train, y_train, sample_weight=sample_weight)
            # x_labels = X.columns
            regression_coefficient_df = pd.DataFrame(index=X.columns, columns=['reg_coeff'])
            regression_coefficient_df['reg_coeff'] = regression.coef_
            # print('intercept:', regression.intercept_, dataset_name)
            results['reg_coeff_' + str(dataset_name) + 'k=' + str(i)] = regression_coefficient_df
            results.to_csv(str(private_folder_path) + 'results.csv')  # save to file
            predictions = regression.predict(X_test)
            predictions = pd.DataFrame(list(zip(y_test.index, predictions)),
                                       columns=['date', str(dataset_name) + ' k=' + str(i)])
            predictions = predictions.set_index('date')
            predictions = out_of_bound_correction(predictions, target_scale_bounds_normalized)

            prediction_results = prediction_results.join(predictions)
            l2_loss = (y_test - predictions[str(dataset_name) + ' k=' + str(i)]) ** 2
            mean_l2_loss_for_one_fold = l2_loss.mean(axis=0)
            # print('L2 loss ' + str(dataset_name) + 'k=' + str(i), ': ', mean_l2_loss_for_one_fold)
            cross_validation_loss_list.append(mean_l2_loss_for_one_fold)
    cross_validation_loss = np.mean(cross_validation_loss_list)
    print('cross_validation_loss: ', cross_validation_loss)
    return prediction_results