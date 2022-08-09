import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from config import checkpoint_path, n_scms, plots_path


def boxplot_from_df(regret_of_setting_df, x_label, y_label, save_name):
    n_settings = regret_of_setting_df.shape[1]
    data_values = np.zeros((n_settings * n_scms))
    for column_idx, column in enumerate(regret_of_setting_df.columns):
        data_values[column_idx * n_scms:(1 + column_idx) * n_scms] = regret_of_setting_df[column]

    data_settings = np.zeros((n_settings * n_scms))
    for i, setting in enumerate(regret_of_setting_df.columns):
        data_settings[i * n_scms:(1 + i) * n_scms] = setting * np.ones(n_scms)

    data_all = np.hstack((data_settings.reshape(-1, 1), data_values.reshape(-1, 1)))
    data_all = pd.DataFrame(data_all, columns=[x_label, y_label])

    ax = sns.boxplot(x=x_label, y=y_label, data=data_all)
    # ax = sns.swarmplot(x=x_label, y=y_label, data=data_all, color=".25")

    plt.tight_layout()
    # ave to file
    plot_path = plots_path + save_name + '.png'
    plt.savefig(plot_path)
    print('saved to:', plot_path)
    plt.show()
    plt.close()


def main():
    # file_name = 'alpha_regret_list_over_simulation_study'
    # x_label = r'$\alpha$'
    # y_label = 'average daily regret'
    # save_name = 'alpha2'
    # setting_loc = 4

    file_name = 'n_ini_obs2'
    x_label = 'number of initial observations'
    y_label = 'average daily regret'
    setting_loc = 1
    save_name = file_name

    # load
    # regret_list_over_simulation_study
    with open(checkpoint_path + str(0) + file_name + '_regret_list_over_simulation_study.pickle', 'rb') as f:
        regret_list_over_simulation_studies, simulation_studies = pickle.load(f)

    for regret_loss, setting in zip(regret_list_over_simulation_studies, simulation_studies):
        regret_loss = np.array(regret_loss)
        regret = regret_loss[:, 0]
        loss = regret_loss[:, 1]
        mean_regret = np.mean(regret)
        mean_loss = np.mean(loss)
        # print('\nsetting:', setting,
        #       '\nmean_regret:', mean_regret,
        #       '\nmean_loss', mean_loss,
        #       '\nregret', regret,
        #       '\nloss', loss, '\n')

    """ plot regret vs setting"""
    # iterate over setting of one var
    settings = []
    regret_of_setting_df = []
    regret_of_setting_95ile = []
    cost_of_setting = []
    for simulation_study, regret_list_over_simulation_studies in zip(simulation_studies,
                                                                     regret_list_over_simulation_studies):
        settings.append(simulation_study[setting_loc])
        regret_over_scm = np.zeros(n_scms)
        cost_over_scm = np.zeros(n_scms)
        for scm_i in range(n_scms):
            regret_over_scm[scm_i] = np.mean(regret_list_over_simulation_studies[scm_i][0])
            cost_over_scm[scm_i] = regret_list_over_simulation_studies[scm_i][1]
        regret_of_setting_df.append(regret_over_scm)
        # regret_of_setting_95ile.append(np.percentile(a=regret_over_scm, q=95))
        cost_of_setting.append(cost_over_scm)
    regret_of_setting_df = pd.DataFrame(np.array(regret_of_setting_df).T, columns=settings)

    boxplot_from_df(
        regret_of_setting_df,
        x_label=x_label,
        y_label=y_label,
        save_name=save_name
    )


main()
