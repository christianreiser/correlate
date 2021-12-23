import math
from datetime import datetime

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config import add_all_yesterdays_features_on, out_of_bound_correction_on, target_label, \
    sample_weights_on, pca_on, autocorrelation_on, histograms_on
from data_cleaning_and_imputation import drop_attributes_with_missing_values, drop_days_before__then_drop_col, \
    drop_days_with_missing_values


def histograms(df, save_path):
    if histograms_on:
        for attribute in df.columns:
            print('histogram:', attribute)

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
    plt.plot(sample_weight[::-1])
    plt.xlabel("Days ago")
    plt.ylabel("Regression Sample Weight")
    plt.title('max(14.7498 - 13.2869 * (x+30) ** 0.0101585,0). \nReaches zero after 82 ~years.')
    plt.savefig('/home/chrei/PycharmProjects/correlate/plots/' + str('RegressionSampleWeight'))
    plt.close('all')
    return sample_weight


def dataset_creation(df):
    df_longest = drop_attributes_with_missing_values(df)
    df_2019_09_08 = drop_days_before__then_drop_col(df, last_day_to_drop='2019-09-08')
    df_widest = drop_days_with_missing_values(df, add_all_yesterdays_features_on)
    return df_longest, df_2019_09_08, df_widest


def normalization(df_not_normalized, min_max):
    # std normalization preprocessing
    df_mean = df_not_normalized.mean()
    df_std = df_not_normalized.std()
    target_mean = df_mean[target_label]
    target_std = df_std[target_label]
    df_normalized = (df_not_normalized - df_mean) / df_std  # built in normalization not used
    print('target_mean:', target_mean)
    print('target_std:', target_std)
    target_scale_bounds_normalized = [(min_max[target_label][0] - df_mean[target_label]) / df_std[target_label],
                                      (min_max[target_label][1] - df_mean[target_label]) / df_std[target_label]]
    return df_normalized, df_not_normalized, target_scale_bounds_normalized, target_mean, target_std, df_mean, df_std


def pca_function(df):
    if pca_on:
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


def autocorrelation(df):
    if autocorrelation_on:
        target_df = df[target_label]

        target_df = drop_days_where_mood_was_tracked_irregularly(target_df)

        # Autocorrelation max lags
        plot_acf(target_df, title=str(target_label) + ' autocorrelation with 95% confidence interval',
                 lags=target_df.shape[0] - 1,
                 alpha=.05, zero=False)
        plt.savefig(
            '/home/chrei/PycharmProjects/correlate/plots/autocorrelation/autocorrelation_' + str(
                target_df.shape[0] - 1) + 'lags_' + str(target_label))

        # Autocorrelation max/2
        # lags
        plot_acf(target_df, title=str(target_label) + ' autocorrelation with 95% confidence interval',
                 lags=math.floor(target_df.shape[0] / 2),
                 alpha=.05, zero=False)
        plt.savefig(
            '/home/chrei/PycharmProjects/correlate/plots/autocorrelation/autocorrelation_' + str(
                math.floor(target_df.shape[0] / 2)) + 'lags_' + str(target_label))

        # Autocorrelation 50 lags
        plot_acf(target_df, title=str(target_label) + ' autocorrelation with 95% confidence interval',
                 lags=50,
                 alpha=.05, zero=False)
        plt.savefig(
            '/home/chrei/PycharmProjects/correlate/plots/autocorrelation/autocorrelation_050lags_' + str(target_label))

        # partial Autocorrelation max lags
        plot_pacf(target_df, lags=math.floor(target_df.shape[0] / 2)-1, alpha=.05, zero=False,
                  title=str(target_label) + ' partial autocorrelation with 95% confidence interval')
        plt.savefig(
            '/home/chrei/PycharmProjects/correlate/plots/autocorrelation/partial_autocorrelation_' + str(
                math.floor(target_df.shape[0] / 2)-1) + 'lags_' + str(
                target_label))

        # partial Autocorrelation 25 lags
        plot_pacf(target_df, lags=25, alpha=.05, zero=False,
                  title=str(target_label) + ' partial autocorrelation with 95% confidence interval')
        plt.savefig(
            '/home/chrei/PycharmProjects/correlate/plots/autocorrelation/partial_autocorrelation_025lags_' + str(
                target_label))
