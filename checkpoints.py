import pickle

from config import checkpoint_path
from data_generation import generate_stationary_scm


def save_checkpoint(ts_measured_actual, was_intervened, ts_generated_actual, ts_generated_optimal, regret_list,
                    random_seed, random_state, coeff, min_coeff, sim_study_input):
    # save input data to file via pickle at checkpoint_path
    # import os
    # filename = os.path.abspath(checkpoint_path)
    with open(checkpoint_path + 'run.pkl', 'wb') as f:
        pickle.dump(
            [ts_measured_actual, was_intervened, ts_generated_actual, ts_generated_optimal, regret_list, random_seed,
             random_state, coeff, min_coeff, sim_study_input], f)


def load_checkpoint(n_measured_links, n_vars_measured, n_vars_all, labels_strs):
    # load input data from file via pickle at checkpoint_path
    # import os
    # filename = os.path.abspath(checkpoint_path)
    with open(checkpoint_path + 'run.pkl', 'rb') as f:
        ts_measured_actual, was_intervened, ts_generated_actual, ts_generated_optimal, regret_list, random_seed, random_state, coeff, min_coeff, sim_study_input = pickle.load(
            f)
    print('WARNING: loaded checkpoint')
    scm, edgemarks_true, effect_sizes_true = generate_stationary_scm(coeff, min_coeff, random_seed, random_state,
                                                                     n_measured_links, n_vars_measured, n_vars_all,
                                                                     labels_strs)
    setting, random_seed, random_state = sim_study_input
    return ts_measured_actual, was_intervened, ts_generated_actual, scm, ts_generated_optimal, regret_list, setting, random_seed, random_state
