from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config import *
from correlation import corr_coefficients_and_p_values
from data_cleaning_and_imputation import *
from fully_connected import fully_connected_nn_prediction
from helper import histograms, drop_days_where_mood_was_tracked_irregularly, dataset_creation
from linear_regression import multiple_linear_regression_ensemble


def main():
    # load data
    df = pd.read_csv('/home/chrei/code/quantifiedSelfData/daily_summaries_all.csv', index_col=0)

    # histograms
    histograms(df, save_path='/home/chrei/PycharmProjects/correlate/plots/distributions/')

    # cleaning and imputation
    df = data_cleaning_and_imputation(df, target_label, add_all_yesterdays_features, add_yesterdays_target_feature,
                                      add_ereyesterdays_target_feature)
    # autocorrelation
    autocorrelation(df)

    # correlation and p value
    results = corr_coefficients_and_p_values(df, target_label)

    # normalization
    df, target_bounds_normalized, target_mean, target_std, df_mean, df_std = normalization(df)

    # dataset_creation
    df_longest, df_2019_09_08, df_widest = dataset_creation(df)

    # PCA
    pca_function(df_2019_09_08)

    # multiple regression
    multiple_linear_regression_ensemble(df=df, df_longest=df_longest, df_2019_09_08=df_2019_09_08, df_widest=df_widest,
                                        results=results,
                                        target_mean=target_mean,
                                        target_std=target_std,
                                        target_bounds_normalized=target_bounds_normalized, df_mean=df_mean,
                                        df_std=df_std)

    # NN
    fully_connected_nn_prediction(df_widest)


def normalization(df):
    # std normalization preprocessing
    df_mean = df.mean()
    df_std = df.std()
    target_mean = df_mean[target_label]
    target_std = df_std[target_label]
    df = (df - df_mean) / df_std  # built in normalization not used
    print('target_mean:', target_mean)
    print('target_std:', target_std)
    target_bounds_normalized = [(target_scale_bounds[0] - df_mean[target_label]) / df_std[target_label],
                                (target_scale_bounds[1] - df_mean[target_label]) / df_std[target_label]]
    return df, target_bounds_normalized, target_mean, target_std, df_mean, df_std


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

        # Autocorrelation 654 lags
        plot_acf(target_df, title=str(target_label) + ' autocorrelation with 95% confidence interval', lags=654,
                 alpha=.05)
        plt.savefig(
            '/home/chrei/PycharmProjects/correlate/plots/autocorrelation/autocorrelation_650lags_' + str(target_label))

        # Autocorrelation 250 lags
        plot_acf(target_df, title=str(target_label) + ' autocorrelation with 95% confidence interval', lags=250,
                 alpha=.05)
        plt.savefig(
            '/home/chrei/PycharmProjects/correlate/plots/autocorrelation/autocorrelation_250lags_' + str(target_label))

        # Autocorrelation 50 lags
        plot_acf(target_df, title=str(target_label) + ' autocorrelation with 95% confidence interval', lags=50,
                 alpha=.05)
        plt.savefig(
            '/home/chrei/PycharmProjects/correlate/plots/autocorrelation/autocorrelation_050lags_' + str(target_label))

        # partial Autocorrelation 326 lags
        plot_pacf(target_df, lags=326, alpha=.05,
                  title=str(target_label) + ' partial autocorrelation with 95% confidence interval')
        plt.savefig(
            '/home/chrei/PycharmProjects/correlate/plots/autocorrelation/partial_autocorrelation_326lags_' + str(
                target_label))

        # partial Autocorrelation 25 lags
        plot_pacf(target_df, lags=25, alpha=.05,
                  title=str(target_label) + ' partial autocorrelation with 95% confidence interval')
        plt.savefig(
            '/home/chrei/PycharmProjects/correlate/plots/autocorrelation/partial_autocorrelation_025lags_' + str(
                target_label))


main()
