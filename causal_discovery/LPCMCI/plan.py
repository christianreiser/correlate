from data_generation import data_generator, generate_stationary_scm, measure

print('import...')
import pickle

import numpy as np
import pandas as pd

from causal_discovery.LPCMCI.observational_discovery import observational_causal_discovery, get_measured_labels
from causal_discovery.interventional_discovery import get_independencies_from_interv_data
from config import target_label, verbosity_thesis, n_vars_measured, coeff, \
    min_coeff, n_vars_all, n_ini_obs, n_mixed, nth, random_seed, tau_max
from intervention_proposal.propose_from_eq import drop_unintervenable_variables, find_most_optimistic_intervention
from intervention_proposal.target_eqs_from_pag import plot_graph, compute_target_equations

"""



"""
"""
main challenges to get algo running:
1. 1 -> 1   Modify data generator to start from last sample 10. june
1. 1 -> 2   intervene in datagenerator                      14->15. june
2. 5 -> 7   Find optimistic intervention in lpcmci graph    9. june
5. 3 -> 1   Lpcmci doesn't use data of a variable if it was intervened upon when calculating its causes 
3. 5 -> 2   Orient edges with interventional data           22 june -> 21. june
                ini complete graph
                    mby similar as 'link_list = ' 
                for each intervened var do CI tests and remove edges
4. 3 -> 2   Initialize lpcmci with above result at all inis 28. june -> 21. june
                (mby replace 'link_list = ' occurrences)            

further TODOs
0. 1->0.3   get negative link colors in lpcmci -> 28.june
1. 2        compute optimal intervention from SCM (for ground truth)6. july
2. 2        calculate regret 11. july
3. 5        set up simulation study 19. july
4. 5        interpret results 27. july
4.5 opt     stochastic intervention? / multiple interventions?
5. 40       write 5. oct

-> 27 coding days + 40 writing days = 57 days = 11.5 weeks = 3 months (optimistic guess) 
    => 3.75 with phinc => end of september

-> 75 coding days + 60 writing days = 135 days = 22 weeks = 5.5 months (guess delay factor: 2.8x coding, 1.5x writing) 
    => 7 with phinc => end of end of year
    
opportunities for computational speedup:
X parallelize
- recycle residuals from lpcmci
x don't run lpcmci when there is no intervention coming
- orient ambiguities towards target var to reduce number of possible graphs
x prune weak links
- instead of append and r_ ini array and fill
"""


def get_intervention_value(var_name, intervention_coeff, ts_measured_actual):
    ts_measured_actual = pd.DataFrame(ts_measured_actual)
    intervention_value = 0  # ini
    intervention_idx = var_name[2:]  # 'u_0' -> '0'
    intervention_var_measured_values = ts_measured_actual[intervention_idx]

    # get 90th percentile of intervention_var_measured_values
    if intervention_coeff > 0:
        intervention_value = np.percentile(intervention_var_measured_values, np.random.uniform(75, 95,
                                                                                               size=1))  # todo is a bit exploration vs exploitation
    elif intervention_coeff < 0:
        intervention_value = np.percentile(intervention_var_measured_values, np.random.uniform(5, 25, size=1))
    else:
        ValueError("intervention_coeff must be positive or negative")
    return intervention_value


def load_eq():
    # load target_ans_per_graph_dict and graph_combinations from file via pickle
    with open('/home/chrei/PycharmProjects/correlate/intervention_proposal/target_eq_simulated.pkl', 'rb') as f:
        target_eq = pickle.load(f)
    with open('/home/chrei/PycharmProjects/correlate/intervention_proposal/graph_combinations_simulated.pkl',
              'rb') as f:
        graph_combinations = pickle.load(f)
    print("attention: target_eq and graph_combinations loaded from file")
    return target_eq, graph_combinations


def find_optimistic_intervention(graph_edgemarks, graph_effect_sizes, labels, ts):
    """
    Optimal control to find the most optimistic intervention.
    """
    # get target equations from graph
    target_eq, graph_combinations = compute_target_equations(
        val_min=graph_effect_sizes,
        graph=graph_edgemarks,
        var_names=labels)

    # load eq instead of calculating them
    # target_eq, graph_combinations = load_eq()

    # remove unintervenable variables
    target_eqs_intervenable = drop_unintervenable_variables(target_eq)

    # get optimal intervention
    largest_abs_coeff, best_intervention_var_name, most_optimistic_graph_idx, intervention_coeff = find_most_optimistic_intervention(
        target_eqs_intervenable)

    # if intervention was found
    if best_intervention_var_name is not None:

        # most optimistic graph
        most_optimistic_graph = graph_combinations[most_optimistic_graph_idx]

        # plot most optimistic graph
        if verbosity_thesis > 0:
            plot_graph(graph_effect_sizes, most_optimistic_graph, labels, 'most optimistic')

        intervention_value = get_intervention_value(best_intervention_var_name, intervention_coeff, ts)
    # if intervention was not found
    else:
        intervention_value = None
    return best_intervention_var_name, intervention_value


def obs_or_intervene(
        n_mixed,
        nth):
    """
    first n_ini_obs samples are observational
    then for n_mixed samples, very nth sample is an intervention
    false: observation
    true: intervention
    """
    # is_obs = np.zeros(n_ini_obs).astype(bool)
    is_mixed = np.zeros(n_mixed).astype(bool)
    for i in range(len(is_mixed)):
        if i % nth == 0:
            is_mixed[i] = True
        else:
            is_mixed[i] = False
    # is_intervention_list = np.append(is_obs, is_mixed)
    return is_mixed


def get_last_outcome(ts_measured_actual):
    """
    in the last sample of ts_measured_actual get value of the target_label
    """
    outcome_last = ts_measured_actual[-1][target_label]  # todo check if it works
    return outcome_last


def get_edgemarks_and_effect_sizes(scm):
    # ini edgemarks ndarray of size (n_vars, n_vars, tau_max)
    edgemarks = np.full([n_vars_all, n_vars_all, tau_max + 1], '', dtype="U3")

    # ini effect sizes ndarray of size (n_vars, n_vars, tau_max)
    effect_sizes = np.zeros((n_vars_all, n_vars_all, tau_max + 1))

    # iterate over all links in scm
    for affected_var in range(len(scm)):
        # get incoming links on affected var
        affected_var_incoming_links = scm[affected_var]
        # for each incoming links on affected var
        for incoming_link in affected_var_incoming_links:
            # int of causing var
            causal_var = incoming_link[0][0]
            # int of tau with minus
            tau = incoming_link[0][1]
            # effect size
            effect_size = incoming_link[1]

            edgemarks[affected_var, causal_var, -tau] = '<--'
            edgemarks[causal_var, affected_var, -tau] = '-->'
            effect_sizes[affected_var, causal_var, -tau] = effect_size
            effect_sizes[causal_var, affected_var, -tau] = effect_size
    return edgemarks, effect_sizes


def store_intervention(was_intervened, intervention_variable, n_samples):
    """
    add data to boolean array of measured variables indicating if they were intervened upon
    input: requires that intervention_variable is a string of the form 'char char int' e.g. 'u_0'
    """

    new_series = pd.Series(np.zeros(n_vars_measured, dtype=bool), index=was_intervened.columns)

    # if intervened
    if intervention_variable is not None:
        # get ind
        intervention_idx = intervention_variable[2:]
        # mark intervened var
        new_series[intervention_idx] = True

    # append new_series to was_intervened
    for i in range(n_samples):
        was_intervened = was_intervened.append(new_series, ignore_index=True)

    # # save was_intervened dataframe to file
    # import os
    # filename = os.path.abspath("./tmp_was_intervened.dat")
    # was_intervened.to_csv(filename, index=False)
    return was_intervened


def main():
    # generate stationary scm
    scm, original_graph = generate_stationary_scm(coeff, min_coeff)
    # get ground truth. Causal discovery on scm

    # ini ts
    ts_generated_actual = np.zeros((0, n_vars_all))
    ts_generated_optimal = np.zeros((0, n_vars_all))

    regret_list = []

    is_intervention_list = obs_or_intervene(
        n_mixed=n_mixed,
        nth=nth)  # 500 obs + 500 with every 4th intervention
    n_samples = 10  # len(is_intervention_list) # todo 1

    measured_labels, measured_label_to_idx = get_measured_labels()

    """ observe first 500 samples"""
    # generate observational data
    ts_df = data_generator(
        scm=scm,
        intervention_variable=None,
        intervention_value=None,
        ts_old=ts_generated_actual,
        random_seed=random_seed,
        n_samples=n_ini_obs[0],
    )

    # measure new data
    ts_measured_actual = measure(ts_df, obs_vars=measured_labels)

    # ini keep track of where the intervention is
    was_intervened = pd.DataFrame(np.zeros((n_ini_obs[0], n_vars_measured), dtype=bool), columns=measured_labels)

    # get ground truth. Causal discovery on scm
    edgemarks_true, effect_sizes_true = get_edgemarks_and_effect_sizes(scm)

    """ loop: causal discovery, planning, intervention """
    for is_intervention_idx in range(len(is_intervention_list)):
        is_intervention = is_intervention_list[is_intervention_idx]

        # if intervention is scheduled
        if is_intervention:

            # causal discovery
            # interventional discovery
            independencies_from_interv_data = get_independencies_from_interv_data(
                ts_measured_actual.copy(),
                was_intervened
            )
            # observational discovery
            pag_effect_sizes, pag_edgemarks = observational_causal_discovery(
                df=ts_measured_actual.copy(),
                was_intervened=was_intervened.copy(),
                external_independencies=independencies_from_interv_data,
                measured_label_to_idx=measured_label_to_idx
            )
            # pag_effect_sizes, pag_edgemarks, var_names = load_results(name_extension='simulated')

            # get interventions
            # actual intervention
            intervention_variable, intervention_value = find_optimistic_intervention(
                pag_edgemarks.copy(),
                pag_effect_sizes.copy(),
                measured_labels,
                ts_measured_actual
            )
            # optimistic intervention on true scm # todo don't need to compute var, sign of value (and value) again
            intervention_var_optimal, intervention_value_optimal = find_optimistic_intervention(
                edgemarks_true.copy(),
                effect_sizes_true.copy(),
                labels=n_vars_all,
                ts=ts_df
            )

            # keep track of where the intervention is
            was_intervened = store_intervention(was_intervened, intervention_variable, n_samples)

        # if no intervention is scheduled
        else:
            intervention_variable = None
            intervention_value = None
            intervention_var_optimal = None
            intervention_value_optimal = None
            was_intervened = store_intervention(was_intervened, intervention_variable, n_samples)

        # intervene as proposed and generate new data
        # actual
        ts_new_actual = data_generator(
            scm=scm,
            intervention_variable=intervention_variable,
            intervention_value=intervention_value,
            ts_old=ts_generated_actual,
            random_seed=random_seed,
            n_samples=n_samples,
        )
        # optimal
        ts_new_optimal = data_generator(
            scm=scm,
            intervention_variable=intervention_var_optimal,
            intervention_value=intervention_value_optimal,
            ts_old=ts_generated_actual,
            random_seed=random_seed,
            n_samples=n_samples,
        )



        # append new (measured) data
        # actual
        ts_generated_actual = np.r_[ts_generated_actual, ts_new_actual] # append new data
        new_measurements = measure(ts_new_actual, obs_vars=measured_labels) # measure new data
        ts_measured_actual = pd.DataFrame(np.r_[ts_measured_actual, new_measurements], columns=measured_labels)
        # optimal
        ts_generated_optimal = pd.DataFrame(np.r_[ts_generated_optimal, ts_new_optimal], columns=measured_labels)

        # regret
        # only if it was an intervention
        if is_intervention:
            regret_list = np.append(regret_list,
                                    abs(get_last_outcome(ts_generated_optimal) - get_last_outcome(ts_measured_actual)))


    regret_sum = sum(regret_list)
    print('regret_sum:', regret_sum)

    print('done')


main()
