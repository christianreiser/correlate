import math

import numpy as np
import pandas as pd
from tqdm import tqdm

from causal_discovery.LPCMCI.observational_discovery import observational_causal_discovery
from causal_discovery.interventional_discovery import get_independencies_from_interv_data
from checkpoints import load_checkpoint, save_checkpoint
from config import target_label, n_vars_measured, coeff, \
    min_coeff, n_vars_all, n_ini_obs, n_mixed, nth, labels_strs, \
    n_samples_per_generation, frac_latents, load_checkpoint_on, verbosity_thesis
from data_generation import data_generator, generate_stationary_scm, measure
from intervention_proposal.get_intervention import find_optimistic_intervention
from regret import compute_regret

"""
Simulation study with Param: # vars, frac, latents, n previous pbs samples, 19. july
Plots: colorscale, latents? 
Anmeldung 
Ask Paul for template or suggest one 
Write key words
3. 5        set up simulation study 
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


def ensure_0_in_measured_labels(measured_labels):
    if 0 not in measured_labels:
        # remove last element of measured_labels
        measured_labels = measured_labels[:-1]
        # add 0 to measured_labels
        measured_labels.append(0)
        measured_labels = np.sort(measured_labels).tolist()
    return measured_labels


def has_measured_cross_dependencies_on_target_var(scm, unmeasured_labels_ints):
    cross_dependencies_on_target_var = []
    for i in range(len(scm[int(target_label)])):
        cross_dependencies_on_target_var.append(scm[int(target_label)][i][0][0])
    # iterate through target_causes and drop if value is 0
    cross_dependencies_on_target_var = [x for x in cross_dependencies_on_target_var if x != 0]
    cross_dependencies_on_target_var = [x for x in cross_dependencies_on_target_var if x not in unmeasured_labels_ints]

    if len(cross_dependencies_on_target_var) < 1:
        return False
    else:
        return True


def get_measured_labels(n_vars_all, random_state, frac_latents, scm):
    """ get measured and unmeasured vars. if there is no measured cross-dependency on target var, resample"""
    unmeasured_labels_ints = None  # ini
    measured_labels = None  # ini
    all_labels_ints = range(n_vars_all)
    cross_dependencies_on_target_var = False
    while not cross_dependencies_on_target_var:
        measured_labels = np.sort(random_state.choice(all_labels_ints,  # e.g. [1,4,5,...]
                                                      size=math.ceil(
                                                          (1. - frac_latents) *
                                                          n_vars_all),
                                                      replace=False)).tolist()
        measured_labels = ensure_0_in_measured_labels(measured_labels)

        # get unmeasured labels
        unmeasured_labels_ints = []
        for x in all_labels_ints:
            if x not in measured_labels:
                unmeasured_labels_ints.append(x)

        # if there is no cross dependency on target var, resample latents
        cross_dependencies_on_target_var = has_measured_cross_dependencies_on_target_var(scm, unmeasured_labels_ints)
        if not cross_dependencies_on_target_var:
            print("no cross dependency on target var, resampling latents")  # todo remove afte check

    unmeasured_labels_strs = [str(x) for x in unmeasured_labels_ints]

    # measured_labels to strings
    measured_labels = [str(x) for x in measured_labels]

    """ key value map of label to index """
    measured_label_as_idx = {label: idx for idx, label in enumerate(measured_labels)}

    # variables that can't be intervened upon. They are the target var and the unobserved vars
    unintervenable_vars = [target_label] + unmeasured_labels_strs

    return measured_labels, measured_label_as_idx, unmeasured_labels_strs, unintervenable_vars


def obs_or_intervene():
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


def store_interv(was_intervened, intervention_variable, n_samples):
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


def simulation_study_with_one_scm(sim_study_input):
    setting, random_seed, random_state = sim_study_input

    # generate stationary scm
    scm, edgemarks_true, effect_sizes_true = generate_stationary_scm(coeff, min_coeff, random_seed, random_state)

    # variable randomly decide which variables are measured vs latent
    measured_labels, measured_label_as_idx, unmeasured_labels_strs, unintervenable_vars = get_measured_labels(
        n_vars_all, random_state, frac_latents, scm)

    # ini ts
    ts_generated_actual, ts_generated_optimal = np.zeros((0, n_vars_all)), np.zeros((0, n_vars_all))

    # ini var that keeps track of where the intervention is
    was_intervened = pd.DataFrame(np.zeros((n_ini_obs[0], n_vars_measured), dtype=bool), columns=measured_labels)

    # ini regret
    regret_list = []

    # ini converged_on_optimal
    converged_on_optimal = False
    interv_var, interv_val = None, None

    # schedule when to intervene
    is_intervention_list = obs_or_intervene()  # 500 obs + 500 with every 4th intervention
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
        old_intervention=[None, None],
    )
    if verbosity_thesis > 1:
        print("intervention_variable_optimal: ", intervention_var_optimal_backup, "interv_val_opti: ",
              intervention_value_optimal_backup)

    # measure new data (hide latents)
    ts_measured_actual = measure(ts_generated_actual, obs_vars=measured_labels)

    if load_checkpoint_on:
        # load data from last checkpoint
        ts_measured_actual, was_intervened, ts_generated_actual, scm, ts_generated_optimal, regret_list, setting, random_seed, random_state = load_checkpoint()

    """ loop: causal discovery, planning, intervention """
    # intervene, observe, observe, observe, ...
    for is_intervention_idx in range(len(is_intervention_list)):

        # stop if converged on optimal
        if not converged_on_optimal:

            # intervene or observe var?
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
                # from measured data
                interv_var, interv_val = find_optimistic_intervention(
                    pag_edgemarks.copy(),
                    pag_effect_sizes.copy(),
                    measured_labels,
                    ts_measured_actual[:n_ini_obs[0]],  # only first n_ini_obs samples, to have the same ts as optimal
                    unintervenable_vars,
                    random_seed,
                    old_intervention=[interv_var, interv_val],
                )

                if verbosity_thesis > 1:
                    print("intervention_variable: ", interv_var, "interv_val: ", interv_val)

                # get interv_var_opti from backup backup
                interv_var_opti = intervention_var_optimal_backup
                interv_val_opti = intervention_value_optimal_backup

                # keep track of where the intervention is
                was_intervened = store_interv(was_intervened, interv_var, n_samples)

            # if no intervention is scheduled
            else:
                interv_var, interv_val, interv_var_opti, interv_val_opti = None, None, None, None
                was_intervened = store_interv(was_intervened, interv_var, n_samples)

            """
            intervene as proposed and generate new data.
            Interv might be none
            """
            # actual
            ts_new_actual = data_generator(
                scm=scm,
                intervention_variable=interv_var,
                intervention_value=interv_val,
                ts_old=ts_generated_actual,
                random_seed=random_seed,
                n_samples=n_samples,
                labels=labels_strs,
            )
            # optimal
            ts_new_optimal = data_generator(
                scm=scm,
                intervention_variable=interv_var_opti,
                intervention_value=interv_val_opti,
                ts_old=ts_generated_actual,
                random_seed=random_seed,
                n_samples=n_samples,
                labels=labels_strs,
            )

            # append new (measured) data
            ts_generated_actual = np.r_[ts_generated_actual, ts_new_actual]  # append actual generated data
            new_measurements = measure(ts_new_actual, obs_vars=measured_labels)  # measure new data (drop latent data)
            ts_measured_actual = pd.DataFrame(np.r_[ts_measured_actual, new_measurements], columns=measured_labels)
            # optimal
            ts_generated_optimal = pd.DataFrame(np.r_[ts_generated_optimal, ts_new_optimal], columns=labels_strs)

            """ 
            regret
            """
            # only if it was an intervention
            if is_intervention:
                regret_list, converged_on_optimal = compute_regret(ts_measured_actual, ts_generated_optimal,
                                                                   regret_list)

            """checkpoint"""
            # save data to checkpoint
            save_checkpoint(ts_measured_actual, was_intervened, ts_generated_actual, ts_generated_optimal, regret_list,
                            random_seed, random_state, coeff, min_coeff, sim_study_input)

    regret_sum = sum(regret_list)
    print('converged on optimal intervention. regret_sum:', regret_sum, '\n\n')
    return regret_sum


def main():
    settings = range(43, 100)  # define_settings()
    regret_per_scm_list = np.zeros(len(settings))
    for run in tqdm(range(100)):
        setting = settings[run]
        random_seed = run
        random_state = np.random.RandomState(random_seed)
        regret_per_scm_list[run] = simulation_study_with_one_scm((setting, random_seed, random_state))
    mean_regret = np.mean(regret_per_scm_list)
    var_regret = np.var(regret_per_scm_list)


main()
