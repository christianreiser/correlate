from datetime import datetime

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from config import add_all_yesterdays_features, out_of_bound_correction_on, plot_distributions, target_label, \
    sample_weights_on
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
    df['mood_not_normalized'] = df[target_label] * target_std + target_mean
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


def bound(low, high, value):
    return max(low, min(high, value))


def generate_sample_weights(y_train):
    # sample weight
    sample_weight = []
    for i in range(y_train.size):
        if sample_weights_on:
            sample_weight.append(max(14.7498 - 13.2869 * (i + 30) ** 0.0101585, 0))
        else:
            sample_weight.append(1)
    sample_weight = sample_weight[::-1]  # reverse list
    plt.plot(sample_weight)
    plt.xlabel("Days ago")
    plt.ylabel("Regression Sample Weight")
    plt.title('max(14.7498 - 13.2869 * (x+30) ** 0.0101585,0). \nReaches zero after 82 ~years.')
    plt.savefig('/home/chrei/PycharmProjects/correlate/plots/' + str('RegressionSampleWeight'))
    plt.close('all')
    return sample_weight


def dataset_creation(df):
    df_longest = drop_attributes_with_missing_values(df)
    df_2019_09_08 = drop_days_before__then_drop_col(df, last_day_to_drop='2019-09-08')
    df_widest = drop_days_with_missing_values(df, add_all_yesterdays_features)
    return df_longest, df_2019_09_08, df_widest
