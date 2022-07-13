import itertools

import numpy as np


def define_settings():
    """ # vars, frac_latents, """
    """
    default: , frac=0.3, 
    n_vars=5; 2 and 10
    frac_latents 0.3;  0.0 and 0.6
    n_pre_obs = 200; 0 - 1000
     = 


    """
    # list from 0 to 1000 in steps of 100


    settings_default_and_list = {}
    settings_default_and_list['n_pre_obs'] = [500, [50, 500, 5000, 50000, 500000]]
    settings_default_and_list['frac_latents'] = [0.3, np.arange(0.0, 0.61, 0.2)]
    settings_default_and_list['n_vars'] = [5, np.arange(2, 11, 2)]
    settings_default_and_list['alpha'] = [0.6, np.arange(0.05, 0.96, 0.15)]
    settings_default_and_list['n_samples_per_generation'] = [10, np.arange(0, 51, 10)]

    setting_vars = ['n_pre_obs', 'frac_latents', 'n_vars', 'alpha', 'n_samples_per_generation']
    default_settings = [500,0.3,6,0.6,5]

    all_param_study_settings = []
    for var in settings_default_and_list.keys():
        this_setting_study_list = settings_default_and_list[var][1]
        one_param_study_settings = []
        for setting in this_setting_study_list:
            one_param_setting = []
            for var2 in settings_default_and_list.keys():
                if var2 != var:
                    one_param_setting.append(settings_default_and_list[var2][0])
                else:
                    one_param_setting.append(setting)
            one_param_study_settings.append(np.array(one_param_setting))
        all_param_study_settings.append(np.array(one_param_study_settings))
    all_param_study_settings = np.array(all_param_study_settings)


    settings_n_pre_obs_study = [[50,0.3,6,0.6,5],[500,0.3,6,0.6,5], [5000,0.3,6,0.6,5], [50000,0.3,6,0.6,5], [500000,0.3,6,0.6,5]]



define_settings()
