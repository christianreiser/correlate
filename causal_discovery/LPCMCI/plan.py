import math
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from causal_discovery.LPCMCI.observational_discovery import observational_causal_discovery
from causal_discovery.gen_configs import define_settings
from causal_discovery.interventional_discovery import get_independencies_from_interv_data
from config import target_label, coeff, min_coeff, n_days, nth, checkpoint_path, n_scms
from data_generation import data_generator, generate_stationary_scm, measure
from intervention_proposal.get_intervention import find_optimistic_intervention
from regret import compute_regret, cost_function

"""
  B) 
3. Paul's list from meeting


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


def is_debug():
    import sys

    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        return False
    else:
        v = gettrace()
        if v is None:
            return False
        else:
            return True


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
    then for n_days samples, very nth sample is an intervention
    false: observation
    true: intervention
    """
    is_mixed = np.zeros(n_days).astype(bool)
    for i in range(len(is_mixed)):
        if i % nth == 0:
            is_mixed[i] = True
        else:
            is_mixed[i] = False
    # is_intervention_list = np.append(is_obs, is_mixed)
    return is_mixed


def store_interv(was_intervened, intervention_variable, n_samples_per_generation, n_vars_measured):
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

    # concat new_series to was_intervened
    tmp_data = []
    for i in range(n_samples_per_generation):
        tmp_data.append(new_series)
    was_intervened = pd.concat([was_intervened, pd.DataFrame(tmp_data)], axis=0, ignore_index=True)

    # # save was_intervened dataframe to file
    # import os
    # filename = os.path.abspath("./tmp_was_intervened.dat")
    # was_intervened.to_csv(filename, index=False)
    return was_intervened


def calculate_parameters(n_vars_measured, frac_latents, n_ini_obs):
    n_measured_links = n_vars_measured
    n_vars_all = math.floor((n_vars_measured / (1. - frac_latents)))  # 11
    labels_strs = [str(i) for i in range(n_vars_all)]
    n_ini_obs = int(n_ini_obs)
    return n_measured_links, n_vars_all, labels_strs, n_ini_obs


def simulation_study_with_one_scm(sim_study_input):
    n_ini_obs, n_vars_measured, frac_latents, pc_alpha, n_samples_per_generation = sim_study_input[0]
    random_seed = sim_study_input[1]
    print('setting:', sim_study_input[0], 'random_seed:', sim_study_input[1])

    n_measured_links, n_vars_all, labels_strs, n_ini_obs = calculate_parameters(n_vars_measured,
                                                                                              frac_latents,
                                                                                              n_ini_obs)

    interv_alpha = 0.95

    random_state = np.random.RandomState(random_seed)

    # generate stationary scm
    scm, edgemarks_true, effect_sizes_true, last_of_ts = generate_stationary_scm(coeff, min_coeff, random_seed, random_state,
                                                                     n_measured_links, n_vars_measured, n_vars_all,
                                                                     labels_strs)

    # variable randomly decide which variables are measured vs latent
    measured_labels, measured_label_as_idx, unmeasured_labels_strs, unintervenable_vars = get_measured_labels(
        n_vars_all, random_state, frac_latents, scm)

    # ini ts
    ts_generated_actual = np.zeros((0, n_vars_all))

    # ini var that keeps track of where the intervention is
    was_intervened = pd.DataFrame(np.zeros((n_ini_obs, n_vars_measured), dtype=bool), columns=measured_labels)

    # ini regret
    regret_list = []

    # ini
    interv_var, interv_val, pag_edgemarks, independencies_from_interv_data = None, None, None, None

    # schedule when to intervene
    is_intervention_list = obs_or_intervene()  # 500 obs + 500 with every 4th intervention
    # n_samples = n_samples_per_generation

    """ generate first n_ini_obs samples without intervention"""
    # generate observational data
    ts_generated_actual, health = data_generator(
        scm=scm,
        intervention_variable=None,
        intervention_value=None,
        ts_old=last_of_ts,
        random_seed=random_seed,
        n_samples=n_ini_obs+100,
        labels=labels_strs,
        noise_type='gaussian',
    )
    ts_generated_actual = ts_generated_actual[-n_ini_obs:]
    ts_generated_optimal = ts_generated_actual

    # measure new data (hide latents)
    ts_measured_actual = measure(ts_generated_actual, obs_vars=measured_labels)

    """ loop: causal discovery, planning, intervention """
    # intervene, observe, observe, observe, ...
    for day, is_intervention in enumerate(is_intervention_list):

        # safe all local variables file
        filename = checkpoint_path + 'global_save1.pkl'
        # with open(filename, 'wb') as f:
        #     pickle.dump([day, is_intervention, ts_generated_actual, regret_list,
        #                  interv_val, ts_measured_actual, ts_generated_optimal, regret_list, was_intervened,
        #                  pag_edgemarks, interv_var, is_intervention_list], f)
        # load
        with open(filename, 'rb') as f:
            day, is_intervention, ts_generated_actual, regret_list, interv_val, ts_measured_actual, ts_generated_optimal, regret_list, was_intervened, pag_edgemarks, interv_var, is_intervention_list = pickle.load(
                f)

        # intervene or observe var?
        is_intervention = is_intervention_list[day]

        # if intervention is scheduled
        if is_intervention:

            """
            causal discovery
            """
            # interventional discovery
            independencies_from_interv_data, dependencies_from_interv_data = get_independencies_from_interv_data(
                ts_measured_actual.copy(),
                was_intervened,
                interv_alpha,
                n_ini_obs,
                pag_edgemarks,
                measured_labels,
            )

            # observational discovery
            pag_effect_sizes, pag_edgemarks = observational_causal_discovery(
                df=ts_measured_actual.copy(),
                was_intervened=was_intervened.copy(),
                external_independencies=independencies_from_interv_data,
                external_dependencies=dependencies_from_interv_data,
                measured_label_to_idx=measured_label_as_idx,
                pc_alpha=pc_alpha,
            )
            # pag_effect_sizes, pag_edgemarks, var_names = load_results(name_extension='simulated')

            """ 
            propose intervention
            """
            # from measured data
            interv_var, interv_val = find_optimistic_intervention(
                my_graph=pag_edgemarks.copy(),
                val=pag_effect_sizes.copy(),
                ts=ts_measured_actual,  # only first n_ini_obs samples, to have the same ts as optimal
                unintervenable_vars=unintervenable_vars,
                random_seed_scm=random_seed,
                random_seed_day=day,
                label='actual_data',
                external_independencies=independencies_from_interv_data,
                external_dependencies=dependencies_from_interv_data,
            )

            # from true SCM
            interv_var_opti, interv_val_opti = find_optimistic_intervention(
                edgemarks_true.copy(),
                effect_sizes_true.copy(),
                ts=pd.DataFrame(ts_generated_actual, columns=labels_strs),
                # needed for 1. percentile from mu, std 2. simulation start 3. labels
                unintervenable_vars=unintervenable_vars,
                random_seed_scm=random_seed,
                random_seed_day=day,
                label='true_scm',
                external_independencies=None,
                external_dependencies=None,
            )



        # if no intervention is scheduled
        else:
            interv_var, interv_val, interv_var_opti, interv_val_opti = None, None, None, None

        # keep track of if and where in the ts the intervention is
        was_intervened = store_interv(was_intervened, interv_var, n_samples_per_generation, n_vars_measured)

        """
        intervene as proposed and generate new data.
        Interv might be none
        """
        # actual
        ts_new_actual, health = data_generator(
            scm=scm,
            intervention_variable=interv_var,
            intervention_value=interv_val,
            ts_old=ts_generated_actual,
            random_seed=random_seed,
            n_samples=n_samples_per_generation,
            labels=labels_strs,
            noise_type='gaussian',
        )
        # optimal
        ts_new_optimal, health = data_generator(
            scm=scm,
            intervention_variable=interv_var_opti,
            intervention_value=interv_val_opti,
            ts_old=ts_generated_optimal,
            random_seed=random_seed,
            n_samples=n_samples_per_generation,
            labels=labels_strs,
            noise_type='gaussian',
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
        regret_list = compute_regret(ts_measured_actual, ts_generated_optimal,
                                     regret_list, n_samples_per_generation)

        if interv_val_opti is not None and interv_val is not None:
            print('rdms:', random_seed, '\tday:', day + n_ini_obs, '\tr', format(regret_list[-1], ".3f"), '\t\to var',
                  interv_var_opti, '\to val', format(interv_val_opti, ".3f"), '\t\ta var',
                  interv_var, '\ta val', format(interv_val, ".3f"), '\tind',independencies_from_interv_data, '\tdep', dependencies_from_interv_data)
        elif interv_val_opti is not None and interv_val is None:
            print('rdms:', random_seed, '\tday:', day + n_ini_obs, '\tr', format(regret_list[-1], ".3f"), '\t\to var',
                  interv_var_opti, '\to val', format(interv_val_opti, ".3f"), '\t\ta var',
                  interv_var, '\ta val', interv_val)
        elif interv_val_opti is None and interv_val is not None:
            print('rdms:', random_seed, '\tday:', day + n_ini_obs, '\tr', format(regret_list[-1], ".3f"), '\t\to var',
                  interv_var_opti, '\to val', interv_val_opti, '\t\ta var',
                  interv_var, '\ta val', interv_val)

    regret_sum = sum(regret_list)
    cost = cost_function(regret_list, was_intervened, n_ini_obs)
    print('regret_sum:', regret_sum, '\n\n')
    return [regret_list, cost]


def run_all_experiments():
    # get settings
    all_param_study_settings = define_settings()

    # run parameter studies
    for simulation_study_idx, simulation_study in enumerate(all_param_study_settings):
        regret_list_over_simulation_study = []

        # run one parameter setting
        for one_param_setting in simulation_study:
            regret_list_over_scms = []

            # repeat each parameter setting for 100 randomly sampled scms

            for i_th_scm in tqdm(range(0, n_scms)):  # n people or scms
                ## run experiment ###
                regret_list_over_scms.append(
                    simulation_study_with_one_scm((one_param_setting, i_th_scm)))
                ######################

            regret_list_over_simulation_study.append(regret_list_over_scms)

        # save results of one parameter setting
        with open(checkpoint_path + str(simulation_study_idx) + 'regret_list_over_simulation_study.pickle', 'wb') as f:
            pickle.dump([regret_list_over_simulation_study, simulation_study], f)
    print('all done')


run_all_experiments()
