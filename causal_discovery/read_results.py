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

    if x_label == 'fraction of interventions':
        data_settings = 1/data_settings

    data_all = np.hstack((data_settings.reshape(-1, 1), data_values.reshape(-1, 1)))

    data_all = pd.DataFrame(data_all, columns=[x_label, y_label])

    # ax = sns.boxenplot(x=x_label, y=y_label, data=data_all)
    ax = sns.boxplot(x=x_label, y=y_label, data=data_all, showfliers=False)

    plt.tight_layout()
    # ave to file
    plot_path = plots_path + save_name + '.png'
    plt.savefig(plot_path)
    print('saved to:', plot_path)
    plt.show()
    plt.close()


def get_regret_of_setting_df(file_name):
    # file_name = 'wo-causality'
    x_label = r'$\withIntervDiscov'
    y_label = 'average daily regret'
    save_name = file_name
    setting_loc = 4

    # file_name = 'nth'
    # x_label = 'fraction of interventions'
    # y_label = 'average daily regret'
    # setting_loc = 7
    # save_name = file_name
    #
    # file_name = 'latents'
    # x_label = 'number of latents'
    # y_label = 'average daily regret'
    # setting_loc = 3
    # save_name = file_name
    #
    # file_name = 'n_ini_obs'
    # x_label = 'number initial observations'
    # y_label = 'average daily regret'
    # setting_loc = 1
    # save_name = file_name
    #
    #
    # file_name = 'n_obs_vars'
    # x_label = 'number of observed variables'
    # y_label = 'average daily regret'
    # setting_loc = 2
    # save_name = file_name

    # load
    # regret_list_over_simulation_study
    with open(checkpoint_path + str(0) + file_name + '_regret_list_over_simulation_study.pickle', 'rb') as f:
        regret_list_over_simulation_studies, simulation_studies = pickle.load(f)

    # for regret_loss, setting in zip(regret_list_over_simulation_studies, simulation_studies):
        # regret_loss = np.array(regret_loss)
        # regret = regret_loss[:, 0]
        # loss = regret_loss[:, 1]
        # mean_regret = np.mean(regret)
        # mean_loss = np.mean(loss)
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

    # plot regret_list_over_simulation_studies[0][0] as timeline
    # r = np.zeros((0, 200))
    # for i in range(len(regret_list_over_simulation_studies)):
    #     plt.plot(regret_list_over_simulation_studies[i][0])
    #     # save to file
    #     plot_path = plots_path + 'regret_timeline/regret_timeline_scm_' + str(i) + '.png'
    #     plt.savefig(plot_path)
    #     # plt.close()
    #     # vstack regret_list_over_simulation_studies[0][i][0]
    #     r = np.vstack((r, regret_list_over_simulation_studies[i][0]))
    # # mean across axis 0
    # r = np.mean(r, axis=0)
    # plt.plot(r)
    # plt.show()

    # get mean of each column in regret_of_setting_df
    print('mean_regret_of_setting_df:', regret_of_setting_df.mean(axis=0))

    boxplot_from_df(
        regret_of_setting_df,
        x_label=x_label,
        y_label=y_label,
        save_name=save_name
    )
    return regret_of_setting_df, regret_list_over_simulation_studies


regret_of_setting_df_1, regret_list_over_simulation_studies_1 = get_regret_of_setting_df('wo-causality')
regret_of_setting_df_2, regret_list_over_simulation_studies_2 = get_regret_of_setting_df('wo-interv')
regret_of_setting_df_3, regret_list_over_simulation_studies_3 = get_regret_of_setting_df('w-interv')

diff_1_2 = regret_of_setting_df_1 - regret_of_setting_df_2
diff_2_3 = regret_of_setting_df_2 - regret_of_setting_df_3



# elementwise addition of diff_1_2 and diff_2_3
diff_1_2_3 = diff_1_2 + diff_2_3

# argmax of diff_2_3
print('argmax of diff_1_2:', np.array(diff_1_2).argmax())
print('argmax of diff_2_3:', np.array(diff_2_3).argmax())
print('argmax of diff_1_2_3:', np.array(diff_1_2_3).argmax())


def cumulator(input_list):
    return np.cumsum(input_list)


cum_1 = cumulator(regret_list_over_simulation_studies_1[37][0])
cum_2 = cumulator(regret_list_over_simulation_studies_2[37][0])
cum_3 = cumulator(regret_list_over_simulation_studies_3[37][0])

# plot regret_of_setting_df_1[0], regret_of_setting_df_2[0], regret_of_setting_df_3[0]
plt.plot(cum_1, label='wo-causality')
plt.plot(cum_2, label='wo-interv')
plt.plot(cum_3, label='w-interv')
plt.legend()

plt.show()
pass