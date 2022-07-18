import numpy as np


def define_settings():
    """
    generate settings for simulation study
    """
    settings_default_and_list = {
        # n obs before first intervention
        'n_ini_obs': [500, [50, 500, 5000]],

        # n measured vars
        'n_vars_measured': [5, np.arange(3, 11, 2)],

        # fraction of additional latents
        'frac_latents': [0.3, np.arange(0.0, 0.61, 0.2)],

        # significance threshold to keep an adjacency
        'alpha': [0.7, np.arange(0.05, 0.96, 0.15)],

        # how often in a row should the same intervention be applied?
        'n_samples_per_generation': [10, np.arange(0, 51, 10)]
    }

    # check if settings are valid
    if max(settings_default_and_list['n_vars_measured'][0],max(settings_default_and_list['n_vars_measured'][1])) > 99:
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
            one_param_study_settings.append(np.array(one_param_setting))
        all_param_study_settings.append(np.array(one_param_study_settings))
    print('total_scms in settings:', total_scms*100)

    return all_param_study_settings


define_settings()