import numpy as np

from config import n_scms


def define_settings():
    """
    generate settings for simulation study
    """

    settings_default_and_list = {
        # n obs before first intervention
        'n_ini_obs': [50, [10, 50, 100]],

        # n measured vars
        'n_vars_measured': [5, np.arange(5, 11, 2)],

        # number of additional latents
        'n_latents': [2, np.arange(0, 4, 1)],

        # significance threshold to keep an adjacency
        'alpha': [0.5, np.arange(0.05, 0.96, 0.3)],

        # how often in a row should the same intervention be applied?
        'n_samples_per_generation': [1, np.arange(0, 5, 2)]
    }

    # settings_default_and_list = {
    #     # n obs before first intervention
    #     'n_ini_obs': [200, [200]],
    #
    #     # n measured vars
    #     'n_vars_measured': [5, [5]],
    #
    #     # fraction of additional latents
    ##     'frac_latents': [0.3, [0.3]],
    #
    #     # significance threshold to keep an adjacency
    #     'alpha': [0.5, [0.5]],
    #
    #     # how often in a row should the same intervention be applied?
    #     'n_samples_per_generation': [1, [1]]
    # }

    # check if settings are valid
    if max(settings_default_and_list['n_vars_measured'][0], max(settings_default_and_list['n_vars_measured'][1])) > 99:
        raise ValueError(
            'Config error. n_vars_measured must have <3 digits. or change len(intervention_variable)>2: in data_generator')

    # get default settings
    default_settings = []
    for key, value in settings_default_and_list.items():
        default_settings.append(value[0])

    # generate settings
    all_param_study_settings = [[default_settings]]
    total_scms = 1
    for var in settings_default_and_list.keys():
        # add default setting
        # all_param_study_settings.append([for var in settings_default_and_list.keys()])
        this_setting_study_list = settings_default_and_list[var][1]
        one_param_study_settings = []
        for setting in this_setting_study_list:
            one_param_setting = []
            total_scms += 1
            for var2 in settings_default_and_list.keys():
                if var2 != var:
                    one_param_setting.append(settings_default_and_list[var2][0])
                else:
                    one_param_setting.append(setting)
            one_param_study_settings.append(np.array(one_param_setting, dtype=object))
        all_param_study_settings.append(np.array(one_param_study_settings, dtype=object))
    print('total_scms in settings:', total_scms * n_scms)

    # #ini obs, #vars, n_latents, alpha, n_samples_per_generation
    all_param_study_settings = [[
        np.array([50, 5, 2, 0.5, 1], dtype=object),
    ]]  # todo remove after testing alpha

    return all_param_study_settings
