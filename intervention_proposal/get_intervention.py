import pickle

import numpy as np
import pandas as pd

from config import checkpoint_path, verbosity_thesis
from intervention_proposal.simulate import get_optimistic_intervention_var_via_simulation
from intervention_proposal.target_eqs_from_pag import plot_graph


def get_intervention_value(var_name, intervention_coeff, ts_measured_actual):
    ts_measured_actual = pd.DataFrame(ts_measured_actual)
    intervention_value = 0  # ini
    # if len >2 then there is the u_ prefix
    if len(var_name) > 2:
        intervention_idx = var_name[2:]  # 'u_0' -> '0'
    else:
        intervention_idx = var_name

    intervention_var_measured_values = ts_measured_actual[intervention_idx]

    # get 90th percentile of intervention_var_measured_values
    if intervention_coeff > 0:
        intervention_value = np.percentile(intervention_var_measured_values, np.random.uniform(75, 95,
                                                                                               size=1))  # todo is a bit exploration vs exploitation
    elif intervention_coeff < 0:
        intervention_value = np.percentile(intervention_var_measured_values, np.random.uniform(5, 25, size=1)) # todo possible to use observational data as 50th percentile data?
    else:
        ValueError("intervention_coeff must be positive or negative")
    return intervention_value

def load_eq():
    # load target_ans_per_graph_dict and graph_combinations from file via pickle
    with open(checkpoint_path+'target_eq_simulated.pkl', 'rb') as f:
        target_eq = pickle.load(f)
    with open(checkpoint_path+'graph_combinations_simulated.pkl',
              'rb') as f:
        graph_combinations = pickle.load(f)
    print("attention: target_eq and graph_combinations loaded from file")
    return target_eq, graph_combinations


def find_optimistic_intervention(graph_edgemarks, graph_effect_sizes, labels, ts):
    """
    Optimal control to find the most optimistic intervention.
    """
    largest_abs_coeff, best_intervention_var_name, most_optimistic_graph_idx, largest_coeff, most_optimistic_graph = get_optimistic_intervention_var_via_simulation(
        graph_effect_sizes, graph_edgemarks, labels, ts
    )

    # # get target equations from graph
    # target_eq, graph_combinations = compute_target_equations(
    #     val_min=graph_effect_sizes,
    #     graph=graph_edgemarks,
    #     var_names=labels)
    #
    # # load eq instead of calculating them
    # # target_eq, graph_combinations = load_eq()
    #
    # # remove unintervenable variables
    # target_eqs_intervenable = drop_unintervenable_variables(target_eq)
    #
    # # get optimal intervention
    # largest_abs_coeff, best_intervention_var_name, most_optimistic_graph_idx, intervention_coeff = find_most_optimistic_intervention(
    #     target_eqs_intervenable)
    #
    # # if intervention was found
    if best_intervention_var_name is not None:
    #
    #     # most optimistic graph
    #     most_optimistic_graph = graph_combinations[most_optimistic_graph_idx]

        # plot most optimistic graph
        if verbosity_thesis > 0:
            plot_graph(graph_effect_sizes, most_optimistic_graph, labels, 'most optimistic')

        intervention_value = get_intervention_value(best_intervention_var_name, largest_coeff, ts)
    # if intervention was not found
    else:
        intervention_value = None
    return best_intervention_var_name, intervention_value
