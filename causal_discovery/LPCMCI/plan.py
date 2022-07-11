print('import...')
import pickle

import numpy as np
import pandas as pd

from causal_discovery.LPCMCI.observational_discovery import observational_causal_discovery
from causal_discovery.interventional_discovery import get_independencies_from_interv_data
from config import target_label, n_vars_measured, coeff, \
    min_coeff, n_vars_all, n_ini_obs, n_mixed, nth, labels_strs, \
    n_samples_per_generation, frac_latents, checkpoint_path
from config_helper import get_measured_labels
from data_generation import data_generator, generate_stationary_scm, measure
from intervention_proposal.get_intervention import find_optimistic_intervention


"""
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


def save_checkpoint(ts_measured_actual, was_intervened, ts_generated_actual, ts_generated_optimal, regret_list,
                    random_seed, random_state, coeff, min_coeff):
    # save input data to file via pickle at checkpoint_path
    # import os
    # filename = os.path.abspath(checkpoint_path)
    with open(checkpoint_path + 'run.pkl', 'wb') as f:
        pickle.dump(
            [ts_measured_actual, was_intervened, ts_generated_actual, ts_generated_optimal, regret_list, random_seed,
             random_state, coeff, min_coeff], f)


def load_checkpoint():
    # load input data from file via pickle at checkpoint_path
    # import os
    # filename = os.path.abspath(checkpoint_path)
    with open(checkpoint_path + 'run.pkl', 'rb') as f:
        ts_measured_actual, was_intervened, ts_generated_actual, ts_generated_optimal, regret_list, random_seed, random_state, coeff, min_coeff = pickle.load(
            f)

    scm, edgemarks_true, effect_sizes_true = generate_stationary_scm(coeff, min_coeff, random_seed, random_state)
    return ts_measured_actual, was_intervened, ts_generated_actual, scm, ts_generated_optimal, regret_list


def simulation_study_with_one_scm(setting, random_seed, random_state):
    measured_labels, measured_label_as_idx, unmeasured_labels_strs = get_measured_labels(n_vars_all, random_state,
                                                                                         frac_latents)
    unintervenable_vars = [target_label] + unmeasured_labels_strs

    # generate stationary scm
    scm, edgemarks_true, effect_sizes_true = generate_stationary_scm(coeff, min_coeff, random_seed, random_state)

    # ini ts
    ts_generated_actual = np.zeros((0, n_vars_all))
    ts_generated_optimal = np.zeros((0, n_vars_all))

    # ini keep track of where the intervention is
    was_intervened = pd.DataFrame(np.zeros((n_ini_obs[0], n_vars_measured), dtype=bool), columns=measured_labels)

    # ini regret
    regret_list = []

    # schedule when to intervene
    is_intervention_list = obs_or_intervene(
        n_mixed=n_mixed,
        nth=nth)  # 500 obs + 500 with every 4th intervention
    n_samples = n_samples_per_generation

    """ generate first 500 samples without intervention"""
    # generate observational data
    ts_generated_actual = data_generator(
        scm=scm,
        intervention_variable=None,
        intervention_value=None,
        ts_old=ts_generated_actual,
        random_seed=random_seed,
        n_samples=n_ini_obs[0],
        labels=labels_strs,
    )

    # optimistic intervention on true scm
    intervention_var_optimal_backup, intervention_value_optimal_backup = find_optimistic_intervention(
        edgemarks_true.copy(),
        effect_sizes_true.copy(),
        labels=[str(var_name) for var_name in range(n_vars_all)],
        ts=ts_generated_actual,
        unintervenable_vars=unintervenable_vars,
        random_seed=random_seed,
    )
    print("intervention_variable_optimal: ", intervention_var_optimal_backup, "intervention_value_optimal: ",
          intervention_value_optimal_backup)

    # measure new data (hide latents)
    ts_measured_actual = measure(ts_generated_actual, obs_vars=measured_labels)

    """ loop: causal discovery, planning, intervention """
    for is_intervention_idx in range(len(is_intervention_list)):

        # load data from last checkpoint
        ts_measured_actual, was_intervened, ts_generated_actual, scm, ts_generated_optimal, regret_list = load_checkpoint()

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
                measured_label_to_idx=measured_label_as_idx
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
                ts_measured_actual[:n_ini_obs[0]],  # only first n_ini_obs samples, to have the same ts as optimal
                unintervenable_vars,
                random_seed,
            )
            print("intervention_variable: ", intervention_variable, "intervention_value: ", intervention_value)

            # get intervention_var_optimal from backup backup
            intervention_var_optimal = intervention_var_optimal_backup
            intervention_value_optimal = intervention_value_optimal_backup

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
        new_measurements = measure(ts_new_actual, obs_vars=measured_labels)  # measure new data (drop latent data)
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
            if new_regret != 0:
                continue
            print('new_regret: ', new_regret)
            regret_list = np.append(regret_list, new_regret)

        # save data to checkpoint
        save_checkpoint(ts_measured_actual, was_intervened, ts_generated_actual, ts_generated_optimal, regret_list,
                        random_seed, random_state, coeff, min_coeff)

    regret_sum = sum(regret_list)
    print('regret_sum:', regret_sum)

    print('done')


def main():
    settings = range(0, 1)
    for setting_idx in range(len(settings)):
        setting = settings[setting_idx]
        random_seed = setting_idx
        random_state = np.random.RandomState(random_seed)
        simulation_study_with_one_scm(setting, random_seed, random_state)


main()
