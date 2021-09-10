import json
from datetime import datetime

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from config import add_all_yesterdays_features, out_of_bound_correction_on, plot_distributions, private_folder_path
from data_cleaning_and_imputation import drop_attributes_with_missing_values, drop_days_before__then_drop_col, \
    drop_days_with_missing_values


def histograms(df, save_path):
    if plot_distributions:
        for attribute in df.columns:
            print(attribute)

            sns.set(style="ticks")

            x = df[attribute]  # .to_numpy()

            f, (ax_box, ax_hist) = plt.subplots(2, sharex=True,
                                                gridspec_kw={"height_ratios": (.15, .85)})

            sns.boxplot(x=x, ax=ax_box, showmeans=True)
            sns.histplot(x=x, bins=50, kde=True)

            ax_box.set(yticks=[])
            sns.despine(ax=ax_hist)
            sns.despine()

            plt.savefig(save_path + str(attribute))
            plt.close('all')
            print('')


def plot_prediction_w_ci_interval(df, ci, target_mean, target_std):
    df = df.copy().dropna()
    df.reset_index(level=0, inplace=True)
    df['prediction_not_normalized'] = df['ensemble_prediction'].multiply(target_std).add(target_mean)
    df['mood_not_normalized'] = df['mood'] * target_std + target_mean
    sns.set_theme(style="darkgrid")
    sns.set(rc={'figure.figsize': (11.7, 8.27)})

    sns.pointplot(x="prediction_not_normalized", y="Date", data=df, join=False, color='r',
                  label="prediction_not_normalized")
    sns.pointplot(x="mood_not_normalized", y="Date", data=df, join=False, color='g', label="mood_not_normalized")
    plt.errorbar(df['prediction_not_normalized'], df['Date'],
                 xerr=np.ones(len(df.loc[:, 'Date'])) * ci * target_std)
    # plt.legend(labels=['legendEntry1', 'legendEntry2'])

    red_patch = mpatches.Patch(color='#bb3f3f', label='prediction')
    green_patch = mpatches.Patch(color='#009152', label='ground truth')
    blue_patch = mpatches.Patch(color='#3045ba', label='95% confidence interval')

    plt.legend(handles=[red_patch, green_patch, blue_patch], loc="upper left")

    plt.tight_layout()
    plt.xlim(0.9, 9.1)
    plt.savefig('/home/chrei/PycharmProjects/correlate/plots/predictions', dpi=200)
    plt.close('all')


def prediction_visualization(df, ci, target_mean, target_std):
    df = df.copy().dropna()
    df.reset_index(level=0, inplace=True)
    df['prediction_not_normalized'] = df['ensemble_prediction'].multiply(target_std).add(target_mean)
    df['mood_not_normalized'] = df['mood'] * target_std + target_mean
    sns.set_theme(style="darkgrid")
    sns.set(rc={'figure.figsize': (11.7, 2.5)})

    df = df.head(1)
    sns.pointplot(x="prediction_not_normalized", y="Date", data=df, join=False, color='r',
                  label="prediction_not_normalized")
    plt.errorbar(df['prediction_not_normalized'], df['Date'],
                 xerr=np.ones(len(df.loc[:, 'Date'])) * ci * target_std)
    # plt.legend(labels=['legendEntry1', 'legendEntry2'])

    red_patch = mpatches.Patch(color='#bb3f3f', label='prediction')
    blue_patch = mpatches.Patch(color='#3045ba', label='95% confidence interval')
    # plt.legend(handles=[red_patch,blue_patch], loc="upper left")

    plt.tight_layout()
    plt.xlim(0.9, 9.1)
    plt.savefig('/home/chrei/PycharmProjects/correlate/plots/one_prediction', dpi=200)
    plt.close('all')


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


def out_of_bound_correction(predictions, target_bounds_normalized):
    if out_of_bound_correction_on:
        # correct if prediction is out of bounds
        for day, i in predictions.iterrows():
            prediction = predictions[predictions.columns[0]][day]
            if prediction > target_bounds_normalized[1]:
                print('out_of_bound_correction: predictions[i]: ', prediction, 'target_upper_bound:',
                      target_bounds_normalized[1])
                correction = target_bounds_normalized[1]
                predictions[predictions.columns[0]][day] = correction

            elif prediction < target_bounds_normalized[0]:
                print('out_of_bound_correction: predictions[i]: ', prediction, 'target_lower_bound:',
                      target_bounds_normalized[0])
                correction = target_bounds_normalized[0]
                predictions[predictions.columns[0]][day] = correction
    return predictions


def dataset_creation(df):
    df_longest = drop_attributes_with_missing_values(df)
    df_2019_09_08 = drop_days_before__then_drop_col(df, last_day_to_drop='2019-09-08')
    df_widest = drop_days_with_missing_values(df, add_all_yesterdays_features)
    return df_longest, df_2019_09_08, df_widest


def write_csv_for_phone_visualization(ci95,
                                      ci68,
                                      target_mean,
                                      target_std_dev,
                                      prediction,
                                      scale_bounds,
                                      feature_weights_normalized,
                                      feature_values_normalized,
                                      feature_values_not_normalized):
    last_prediction_date = prediction.dropna().index.array[prediction.dropna().index.array.size - 1]
    prediction = prediction[last_prediction_date]

    # feature weights

    # drop zeros
    feature_weights_normalized = feature_weights_normalized[feature_weights_normalized != 0]
    feature_weights_not_normalized = feature_weights_normalized*target_std_dev


    # feature values
    feature_values_not_normalized = feature_values_not_normalized.loc[last_prediction_date]
    feature_values_normalized = feature_values_normalized.loc[last_prediction_date]
    features_df = pd.DataFrame(index=feature_values_normalized.index,
                               columns=['weights', 'values_normalized', 'values_not_normalized', 'contribution',
                                        'contribution_abs'])
    features_df['weights'] = feature_weights_not_normalized
    features_df['values_not_normalized'] = feature_values_not_normalized
    features_df['values_normalized'] = feature_values_normalized

    features_df['contribution'] = features_df['weights'].multiply(features_df['values_normalized'])
    features_df = features_df.dropna(subset=['weights'])

    features_df['contribution_abs'] = abs(features_df['contribution'])
    features_df = features_df.sort_values(by='contribution_abs', ascending=False)

    # check prediction reverse engineering works
    fake_prediction = target_mean + (features_df['contribution'].sum())*(1.1)

    # prediction values
    prediction_dict = {
        "prediction": round(fake_prediction, 2),  #round(prediction * target_std_dev + target_mean, 2), #
        "ci68": round(ci68 * target_std_dev, 2),
        "ci95": round(ci95 * target_std_dev, 2),
        "scale_bounds": list(np.around(np.array(scale_bounds), 2)),
        "target_mean": round(target_mean, 2),
    }
    with open(str(private_folder_path) + 'phone_io/prediction.json', 'w') as f:
        json.dump(prediction_dict, f)

    feature_data_df = pd.DataFrame(index=features_df.index,
                                   columns=['start_contribution', 'end_contribution', 'positive_effect'])
    previous_end = target_mean
    for i, row in features_df.iterrows():
        feature_data_df.loc[i, 'start_contribution'] = previous_end
        feature_data_df.loc[i, 'end_contribution'] = previous_end + features_df.loc[i, 'contribution']
        if (previous_end <= previous_end + features_df.loc[i, 'contribution']):
            feature_data_df.loc[i, 'positive_effect'] = True
        else:
            feature_data_df.loc[i, 'positive_effect'] = False
            tmp = feature_data_df.loc[i, 'start_contribution']
            feature_data_df.loc[i, 'start_contribution'] = feature_data_df.loc[i, 'end_contribution']
            feature_data_df.loc[i, 'end_contribution'] = tmp
        previous_end = previous_end + features_df.loc[i, 'contribution']
        feature_data_df.loc[i, 'start_contribution'] = round(feature_data_df.loc[i, 'start_contribution'], 2)
        feature_data_df.loc[i, 'end_contribution'] = round(feature_data_df.loc[i, 'end_contribution'], 2)
    print('explained mean: ', previous_end)

    feature_data_df.to_csv(str(private_folder_path) + 'phone_io/feature_data.csv', line_terminator='\r\n')

    print(target_mean, prediction, ci95, ci68, scale_bounds, feature_weights_not_normalized)
