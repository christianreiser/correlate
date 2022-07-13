import numpy as np

from config import target_label, n_samples, n_0_regret


def get_last_outcome(ts_measured_actual):
    """
    in the last n_samples of ts_measured_actual get value of the target_label
    """
    outcome_last = np.array(ts_measured_actual.loc[:, target_label])[-n_samples:]
    return outcome_last


def check_converged_on_optimal(regret_list):
    """
    check if the optimal solution has been reached n_0_regret times in a row
    """
    if len(regret_list) < n_0_regret:
        return False
    else:
        # regret_list = regret_list[-3:] # todo activate
        if np.all(regret_list[-n_0_regret:] == [0] * n_0_regret):
            return True
        else:
            return False

def compute_regret(ts_measured_actual, ts_generated_optimal, regret_list):
    outcome_actual = get_last_outcome(ts_measured_actual)
    outcome_optimal = get_last_outcome(ts_generated_optimal)
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
    print('new_regret: ', new_regret)
    regret_list = np.append(regret_list, new_regret)

    # check if converged on optimal
    converged_on_optimal = check_converged_on_optimal(regret_list)

    return regret_list, converged_on_optimal