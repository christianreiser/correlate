import json

import numpy as np
import pandas as pd

from config import target_label, phone_vis_height_width, survey_value_manipulation
from helper import bound


def write_csv_for_phone_visualization(ci95,
                                      ci68,
                                      target_mean,
                                      target_std_dev,
                                      prediction,
                                      scale_bounds,
                                      feature_weights_normalized,
                                      feature_values_normalized,
                                      feature_values_not_normalized,
                                      min_max):
    last_prediction_date = prediction.dropna().index.array[prediction.dropna().index.array.size - 1]  # date to vis
    prediction = prediction[last_prediction_date]

    # somehow weights_not_notmalized * value_normalized != predction by library
    fake_factor = 1.1  # todo: proper fix

    # drop zeros
    feature_weights_normalized = feature_weights_normalized[feature_weights_normalized != 0]
    feature_weights_not_normalized = feature_weights_normalized * target_std_dev * fake_factor

    # feature values
    feature_values_not_normalized = feature_values_not_normalized.loc[last_prediction_date]
    feature_values_normalized = feature_values_normalized.loc[last_prediction_date]

    # get_features_df
    features_df = get_features_df(feature_values_normalized,
                                  feature_weights_not_normalized,
                                  feature_values_not_normalized,
                                  target_mean,
                                  scale_bounds)

    # write_regression_triangle_chart_file
    write_regression_triangle_chart_file(features_df, min_max)

    # write_gantt_chart_file
    previous_end = write_gantt_chart_file(features_df, scale_bounds)

    # write_prediction_file
    write_prediction_file(previous_end, ci68, ci95, target_std_dev, scale_bounds, target_mean)



def get_features_df(feature_values_normalized, feature_weights_not_normalized, feature_values_not_normalized,
                    target_mean, scale_bounds):
    features_df = pd.DataFrame(
        index=np.concatenate([feature_values_normalized.index.to_numpy(), np.array(['MoodAverage()'])]),
        columns=['weights', 'values_normalized', 'values_not_normalized', 'contribution',
                 'contribution_abs', 'scale_size_time_phone_viz_height'])
    features_df['weights'] = feature_weights_not_normalized
    features_df['values_not_normalized'] = feature_values_not_normalized
    features_df['values_normalized'] = feature_values_normalized

    features_df['contribution'] = features_df['weights'].multiply(features_df['values_normalized'])
    features_df.loc['MoodAverage()', 'contribution'] = target_mean - np.mean(scale_bounds)

    features_df = features_df.dropna(subset=['contribution'])

    features_df['contribution_abs'] = abs(features_df['contribution'])
    features_df = features_df.sort_values(by='contribution_abs', ascending=False)
    return features_df


def write_gantt_chart_file(features_df, scale_bounds):
    gantt_chart_df = pd.DataFrame(index=features_df.index,
                                  columns=['start_contribution', 'end_contribution', 'positive_effect'])
    previous_end = np.mean(scale_bounds)
    for i, row in features_df.iterrows():
        gantt_chart_df.loc[i, 'start_contribution'] = previous_end
        gantt_chart_df.loc[i, 'end_contribution'] = previous_end + features_df.loc[i, 'contribution']
        if previous_end <= previous_end + features_df.loc[i, 'contribution']:
            gantt_chart_df.loc[i, 'positive_effect'] = True
        else:
            gantt_chart_df.loc[i, 'positive_effect'] = False
            tmp = gantt_chart_df.loc[i, 'start_contribution']
            gantt_chart_df.loc[i, 'start_contribution'] = gantt_chart_df.loc[i, 'end_contribution']
            gantt_chart_df.loc[i, 'end_contribution'] = tmp
        previous_end = previous_end + features_df.loc[i, 'contribution']
        gantt_chart_df.loc[i, 'start_contribution'] = round(gantt_chart_df.loc[i, 'start_contribution'], 3)
        gantt_chart_df.loc[i, 'end_contribution'] = round(gantt_chart_df.loc[i, 'end_contribution'], 3)
    print('explained mean: ', previous_end)
    gantt_chart_df.to_csv('/home/chrei/code/insight_me/assets/tmp_phone_io/gantt_chart.csv', line_terminator='\r\n')
    return previous_end


def write_regression_triangle_chart_file(features_df, min_max):
    """
    regression_triangle: weight_value_contribution
    """
    features_df = features_df.drop(['MoodAverage()'], axis=0)
    regression_triangle_chart_df = pd.DataFrame(index=features_df.index,
                                                columns=['mean_x_coord', 'mean_y_coord', 'dosage_coord',
                                                         'response_coord', 'scale_size', 'phone_width_factor'])
    min_max = min_max.T
    regression_triangle_chart_df['scale_size'] = min_max['max'] - min_max['min']
    scale_size_target = min_max['max'][target_label] - min_max['min'][target_label]
    phone_height_factor = phone_vis_height_width[0] / scale_size_target

    for i, row in regression_triangle_chart_df.iterrows():
        regression_triangle_chart_df.loc[i, 'phone_width_factor'] = phone_vis_height_width[1] / \
                                                                    regression_triangle_chart_df.loc[i, 'scale_size']

    regression_triangle_chart_df['mean_y_coord'] = (scale_size_target - (
                min_max['mean'][target_label] - min_max['min'][target_label])) \
                                                   * phone_height_factor
    regression_triangle_chart_df['mean_x_coord'] = (regression_triangle_chart_df['scale_size'] - (
                min_max['max'] - min_max['mean'])) \
                                                   * regression_triangle_chart_df['phone_width_factor']

    for i, row in regression_triangle_chart_df.iterrows():
        regression_triangle_chart_df.loc[i, 'dosage_coord'] = (features_df.loc[i, 'values_not_normalized'] -
                                                               min_max.loc[i, 'min']) * \
                                                              regression_triangle_chart_df.loc[i, 'phone_width_factor']
        regression_triangle_chart_df.loc[i, 'response_coord'] = regression_triangle_chart_df.loc[i, 'mean_y_coord'] - \
                                                                features_df.loc[i, 'contribution'] * phone_height_factor

    regression_triangle_chart_df = regression_triangle_chart_df.drop(['phone_width_factor', 'scale_size'], axis=1)

    # round
    for i, row in features_df.iterrows():
        regression_triangle_chart_df.loc[i, 'mean_x_coord'] = round(regression_triangle_chart_df.loc[i, 'mean_x_coord'],
                                                                    3)
        regression_triangle_chart_df.loc[i, 'mean_y_coord'] = round(regression_triangle_chart_df.loc[i, 'mean_y_coord'],
                                                                    3)
        regression_triangle_chart_df.loc[i, 'dosage_coord'] = round(regression_triangle_chart_df.loc[i, 'dosage_coord'],
                                                                    3)
        regression_triangle_chart_df.loc[i, 'response_coord'] = round(
            regression_triangle_chart_df.loc[i, 'response_coord'], 3)

    regression_triangle_chart_df.to_csv('/home/chrei/code/insight_me/assets/tmp_phone_io/regression_triangle_chart.csv',
                                        line_terminator='\r\n')


def write_prediction_file(previous_end, ci68, ci95, target_std_dev, scale_bounds, target_mean):
    ci95_not_normalized = ci95 * target_std_dev
    ci95 = [
        # math.ceil(
        round(bound(scale_bounds[0], scale_bounds[1], previous_end - ci95_not_normalized), 3),
        # ),math.floor(
        round(bound(scale_bounds[0], scale_bounds[1], previous_end + ci95_not_normalized), 3)
        # )
    ]
    prediction_dict = {
        "prediction": round(previous_end, 3),
        "ci68": round(ci68 * target_std_dev, 3),
        "ci95": ci95,
        "scale_bounds": list(np.around(np.array(scale_bounds), 2)),
        "target_mean": round(target_mean, 3),
    }
    with open('/home/chrei/code/insight_me/assets/tmp_phone_io/prediction.json', 'w') as f:
        json.dump(prediction_dict, f)
