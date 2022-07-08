from data_generation import data_generator, generate_stationary_scm, measure
from intervention_proposal.get_intervention import find_optimistic_intervention

print('import...')

import numpy as np
import pandas as pd

from causal_discovery.LPCMCI.observational_discovery import get_measured_labels, observational_causal_discovery
from causal_discovery.interventional_discovery import get_independencies_from_interv_data
from config import target_label, n_vars_measured, coeff, \
    min_coeff, n_vars_all, n_ini_obs, n_mixed, nth, random_seed, labels_strs

"""



"""
"""
1. if same intervention var should lead to same outcome
2. intervenional discovery doesn;'t work

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


def get_last_outcome(ts_measured_actual, n_samples):
    """
    in the last n_samples of ts_measured_actual get value of the target_label
    """
    outcome_last = np.array(ts_measured_actual.loc[:, target_label])[-n_samples:]
    return outcome_last





def store_intervention(was_intervened, intervention_variable, n_samples):
    """
    add data to boolean array of measured variables indicating if they were intervened upon
    input: requires that intervention_variable is a string of the form 'char char int' e.g. 'u_0'
    """

    new_series = pd.Series(np.zeros(n_vars_measured, dtype=bool), index=was_intervened.columns)

    # if intervened
    if intervention_variable is not None:
        # get ind
        if len(intervention_variable) > 2:
            intervention_idx = intervention_variable[2:]
        else:
            intervention_idx = intervention_variable

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
    scm, edgemarks_true, effect_sizes_true = generate_stationary_scm(coeff, min_coeff)

    # ini ts
    ts_generated_actual = np.zeros((0, n_vars_all))
    ts_generated_optimal = np.zeros((0, n_vars_all))

    # ini regret
    regret_list = []

    # schedule when to intervene
    is_intervention_list = obs_or_intervene(
        n_mixed=n_mixed,
        nth=nth)  # 500 obs + 500 with every 4th intervention
    n_samples = 10  # len(is_intervention_list) # todo 1

    measured_labels, measured_label_to_idx = get_measured_labels()

    """ generate first 500 samples without intervention"""
    # generate observational data
    ts_df = data_generator(
        scm=scm,
        intervention_variable=None,
        intervention_value=None,
        ts_old=ts_generated_actual,
        random_seed=random_seed,
        n_samples=n_ini_obs[0],
        labels=labels_strs,
    )

    # measure new data (hide latents)
    ts_measured_actual = measure(ts_df, obs_vars=measured_labels)

    # ini keep track of where the intervention is
    was_intervened = pd.DataFrame(np.zeros((n_ini_obs[0], n_vars_measured), dtype=bool), columns=measured_labels)

    """ loop: causal discovery, planning, intervention """
    for is_intervention_idx in range(len(is_intervention_list)):
        is_intervention = is_intervention_list[is_intervention_idx]

        # if intervention is scheduled
        if is_intervention:

            """
            causal discovery
            """
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

            """ 
            propose intervention
            """
            # actual intervention
            intervention_variable, intervention_value = find_optimistic_intervention(
                pag_edgemarks.copy(),
                pag_effect_sizes.copy(),
                measured_labels,
                ts_measured_actual
            )

            print("intervention_variable: ", intervention_variable, "intervention_value: ", intervention_value)
            # optimistic intervention on true scm # todo don't need to compute var, sign of value (and value) again
            intervention_var_optimal, intervention_value_optimal = find_optimistic_intervention(
                edgemarks_true.copy(),
                effect_sizes_true.copy(),
                labels=[str(var_name) for var_name in range(n_vars_all)],
                ts=ts_df
            )
            print("intervention_variable_optimal: ", intervention_var_optimal, "intervention_value_optimal: ",
                  intervention_value_optimal)

            # keep track of where the intervention is
            was_intervened = store_intervention(was_intervened, intervention_variable, n_samples)

        # if no intervention is scheduled
        else:
            intervention_variable = None
            intervention_value = None
            intervention_var_optimal = None
            intervention_value_optimal = None
            was_intervened = store_intervention(was_intervened, intervention_variable, n_samples)

        """
        intervene as proposed and generate new data
        """
        # actual
        ts_new_actual = data_generator(
            scm=scm,
            intervention_variable=intervention_variable,
            intervention_value=intervention_value,
            ts_old=ts_generated_actual,
            random_seed=random_seed,
            n_samples=n_samples,
            labels=labels_strs,
        )
        # optimal
        ts_new_optimal = data_generator(
            scm=scm,
            intervention_variable=intervention_var_optimal,
            intervention_value=intervention_value_optimal,
            ts_old=ts_generated_actual,
            random_seed=random_seed,
            n_samples=n_samples,
            labels=labels_strs,
        )

        # append new (measured) data
        # actual
        ts_generated_actual = np.r_[ts_generated_actual, ts_new_actual]  # append new data
        new_measurements = measure(ts_new_actual, obs_vars=measured_labels)  # measure new data
        ts_measured_actual = pd.DataFrame(np.r_[ts_measured_actual, new_measurements], columns=measured_labels)
        # optimal
        ts_generated_optimal = pd.DataFrame(np.r_[ts_generated_optimal, ts_new_optimal], columns=labels_strs)

        """ 
        compute regret
        """
        # only if it was an intervention
        if is_intervention:
            outcome_actual = get_last_outcome(ts_measured_actual, n_samples)
            outcome_optimal = get_last_outcome(ts_generated_optimal, n_samples)
            new_regret = sum(outcome_optimal - outcome_actual)
            print('new_regret: ', new_regret)
            regret_list = np.append(regret_list, new_regret)

    regret_sum = sum(regret_list)
    print('regret_sum:', regret_sum)

    print('done')


main()
