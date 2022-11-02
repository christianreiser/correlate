import numpy as np
import pandas as pd

from config import target_label, verbosity_thesis


def get_last_outcome(ts_measured_actual, n_samples_per_generation):
    """
    in the last n_samples_per_generation of ts_measured_actual get value of the target_label
    """
    outcome_last = np.array(ts_measured_actual.loc[:, target_label])[-n_samples_per_generation:]
    return outcome_last


def compute_regret(ts_measured_actual, ts_generated_optimal, regret_list, n_samples_per_generation, interv_var_optil, interv_var, interv_var_correct_list):
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
    regret_list = np.append(regret_list, new_regret)

    # if interv_var_opti == interv_var then add 1 to ts_interv_var_correct else add 0
    if interv_var_optil == interv_var:
        interv_var_correct_list = np.append(interv_var_correct_list, 1)
    else:
        interv_var_correct_list = np.append(interv_var_correct_list, 0)
    return regret_list, outcome_actual[0], interv_var_correct_list


def test_compute_regret():
    ts_measured_actual = pd.DataFrame(np.array([[1, 3], [4, 6], [7, 9]]), columns=['0', '1'])
    ts_generated_optimal = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['0', '1', '2'])
    regret_list = np.array([])
    n_samples_per_generation = 1
    regret_list = compute_regret(ts_measured_actual, ts_generated_optimal, regret_list, n_samples_per_generation)
    assert regret_list == [0]


def cost_function(regret_list, was_intervened, n_ini_obs):
    """
    compute cost function
    """
    cost_per_observation = 1
    cost_per_intervention = 10
    cost_per_regret = 34  # 3.4*10
    # count number of interventions
    n_interventions = was_intervened.to_numpy().sum()
    # count number of observations
    n_observations = was_intervened.shape[0] - n_interventions
    # compute cost
    sum_regret = sum(regret_list)
    cost = cost_per_observation * n_observations + cost_per_intervention * n_interventions + cost_per_regret * sum_regret
    print(
        'cost', cost, ' = cost_per_observation', cost_per_observation, ' * n_observations', n_observations,
        ' + cost_per_intervention', cost_per_intervention, ' * n_interventions', n_interventions, ' + cost_per_regret',
        cost_per_regret, ' * sum_regret', sum_regret)
    return cost
