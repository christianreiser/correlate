import json

import numpy as np
import pandas as pd


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

    # write_wvc_chart_file
    write_wvc_chart_file(features_df)

    # write_gantt_chart_file
    previous_end = write_gantt_chart_file(features_df, scale_bounds)

    # write_prediction_file
    write_prediction_file(previous_end, ci68, ci95, target_std_dev, scale_bounds, target_mean)

    print(target_mean, prediction, ci95, ci68, scale_bounds, feature_weights_not_normalized)


def get_features_df(feature_values_normalized, feature_weights_not_normalized, feature_values_not_normalized,
                    target_mean, scale_bounds):
    features_df = pd.DataFrame(
        index=np.concatenate([feature_values_normalized.index.to_numpy(), np.array(['average_mood'])]),
        columns=['weights', 'values_normalized', 'values_not_normalized', 'contribution',
                 'contribution_abs'])
    features_df['weights'] = feature_weights_not_normalized
    features_df['values_not_normalized'] = feature_values_not_normalized
    features_df['values_normalized'] = feature_values_normalized

    features_df['contribution'] = features_df['weights'].multiply(features_df['values_normalized'])
    features_df.loc['average_mood', 'contribution'] = target_mean - np.mean(scale_bounds)

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


def write_wvc_chart_file(features_df):
    """
    WVC: weight_value_contribution
    """
    features_df = features_df.drop(['average_mood'], axis=0)
    WVC_chart_df = pd.DataFrame(index=features_df.index,
                                columns=['contribution', 'weight', 'value_today_not_normalized',
                                         'value_today_normalized', 'extrema'])
    WVC_chart_df['contribution'] = features_df['contribution']
    WVC_chart_df['weight'] = features_df['weights']
    WVC_chart_df['value_today_not_normalized'] = features_df['values_not_normalized']
    WVC_chart_df['value_today_normalized'] = features_df['values_normalized']

    for i, row in features_df.iterrows():
        WVC_chart_df.loc[i, 'contribution'] = round(WVC_chart_df.loc[i, 'contribution'], 3)
        WVC_chart_df.loc[i, 'weight'] = round(WVC_chart_df.loc[i, 'weight'], 3)
        WVC_chart_df.loc[i, 'value_today_not_normalized'] = round(WVC_chart_df.loc[i, 'value_today_not_normalized'], 3)
    WVC_chart_df.loc[i, 'value_today_normalized'] = round(WVC_chart_df.loc[i, 'value_today_normalized'], 3)

    # normalize columns
    # for column in WVC_chart_df:
    #     WVC_chart_df[column] = (WVC_chart_df[column] - WVC_chart_df[column].min()) / (
    #                 WVC_chart_df[column].max() - WVC_chart_df[column].min())
    #     WVC_chart_df[column] = (WVC_chart_df[column] ) / (
    #                 WVC_chart_df[column].std())

    # get min max ofr scale bounds
    normalized_df = WVC_chart_df.drop(['value_today_not_normalized'], axis=1)
    WVC_chart_df.loc[WVC_chart_df.index.to_numpy()[0], 'extrema'] = normalized_df.max().max()
    WVC_chart_df.loc[WVC_chart_df.index.to_numpy()[1], 'extrema'] = normalized_df.min().min()

    WVC_chart_df.to_csv('/home/chrei/code/insight_me/assets/tmp_phone_io/wvc_chart.csv', line_terminator='\r\n')


def write_prediction_file(previous_end, ci68, ci95, target_std_dev, scale_bounds, target_mean):
    prediction_dict = {
        "prediction": round(previous_end, 3),
        "ci68": round(ci68 * target_std_dev, 3),
        "ci95": round(ci95 * target_std_dev, 3),
        "scale_bounds": list(np.around(np.array(scale_bounds), 2)),
        "target_mean": round(target_mean, 3),
    }
    with open('/home/chrei/code/insight_me/assets/tmp_phone_io/prediction.json', 'w') as f:
        json.dump(prediction_dict, f)
