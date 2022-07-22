import numpy as np

from config import target_label, regret_convergence_thresh, n_below_regret_thresh, verbosity_thesis


def get_last_outcome(ts_measured_actual, n_samples_per_generation):
    """
    in the last n_samples_per_generation of ts_measured_actual get value of the target_label
    """
    outcome_last = np.array(ts_measured_actual.loc[:, target_label])[-n_samples_per_generation:]
    return outcome_last


def check_converged_on_optimal(regret_list):
    """
    check if the optimal solution has been reached n_0_regret times in a row
    """
    if len(regret_list) < n_below_regret_thresh:
        return False
    else:
        regret_list = regret_list[-n_below_regret_thresh:]
        if np.all(regret_list[-n_below_regret_thresh:] <= [regret_convergence_thresh] * n_below_regret_thresh):
            return True
        else:
            return False

def compute_regret(ts_measured_actual, ts_generated_optimal, regret_list, n_samples_per_generation):
    outcome_actual = get_last_outcome(ts_measured_actual, n_samples_per_generation)
    outcome_optimal = get_last_outcome(ts_generated_optimal, n_samples_per_generation)
    new_regret = sum(outcome_optimal - outcome_actual)
    # if new_regret < 0:
    #     print('outcome_optimal:', outcome_optimal,
    #           '\noutcome_actual:', outcome_actual,
    #           '\nintervention_variable:', interv_var,
    #           '\ninterv_val:', interv_val,
    #           '\nintervention_value_optimal_backup:', intervention_value_optimal_backup,
    #           '\nintervention_var_optimal_backup:', intervention_var_optimal_backup,
    #           '\nintervention_variable:', interv_var)
    #     ValueError("Regret is negative! See prints above")
    if verbosity_thesis >0:
        print('new_regret: ', new_regret)
    regret_list = np.append(regret_list, new_regret)

    # check if converged on optimal
    converged_on_optimal = check_converged_on_optimal(regret_list)

    return regret_list, converged_on_optimal