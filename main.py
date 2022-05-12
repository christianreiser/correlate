from causal_discovery.LPCMCI.experiment import causal_discovery
from config import *
from correlation import corr_coefficients_and_p_values
from data_cleaning_and_imputation import *
from helper import histograms, dataset_creation, pca_function, \
    autocorrelation, normalization
from prediction.fully_connected import fully_connected_nn_prediction
from prediction.linear_regression import multiple_linear_regression_ensemble


def main():
    # load data
    df = pd.read_csv(str(private_folder_path) + 'daily_summaries_compute.csv', index_col=0)

    if survey_value_manipulation:
        df.loc['2021-06-14', 'DistractingScreentime'] = 658

    # histograms
    histograms(df, save_path='/home/chrei/PycharmProjects/correlate/plots/distributions/')

    # cleaning and imputation
    df = data_cleaning_and_imputation(df, target_label, add_all_yesterdays_features_on,
                                      add_yesterdays_target_feature_on,
                                      add_ereyesterdays_target_feature_on)

    min_max = df.agg(['min', 'max', 'mean'])

    # autocorrelation
    autocorrelation(df)

    # correlation and p value
    results = corr_coefficients_and_p_values(df, target_label)

    # normalization
    df, df_not_normalized, target_scale_bounds_normalized, target_mean, target_std, df_mean, df_std = normalization(
        df_not_normalized=df, min_max=min_max)

    # dataset_creation
    df_longest, df_2019_09_08, df_widest = dataset_creation(df)

    # PCA
    pca_function(df_widest)

    # multiple regression
    multiple_linear_regression_ensemble(df=df, df_not_normalized=df_not_normalized, df_longest=df_longest,
                                        df_2019_09_08=df_2019_09_08, df_widest=df_widest,
                                        results=results,
                                        target_mean=target_mean,
                                        target_std=target_std,
                                        target_scale_bounds_normalized=target_scale_bounds_normalized,
                                        min_max=min_max)

    # NN
    fully_connected_nn_prediction(df_widest)

    # causal discovery
    causal_discovery(df)


main()
