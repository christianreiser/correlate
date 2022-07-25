import pandas as pd
from statsmodels.compat.pandas import assert_frame_equal
from config import checkpoint_path
from data_generation import data_generator
from intervention_proposal.get_intervention import lin_f

class TestGetIntervention:
    def test_data_generator(self):
        # ERROR    X[:max_lag] = ts_old[-max_lag:]
        # ValueError: could not broadcast input array from shape (500,5) into shape (0,5)
        scm = {0: [((3, 0), 0.47058655898587115, lin_f)],
               1: [((2, 0), -0.04401074099467584, lin_f)],
               2: [((4, 0), 0.029253068218103893, lin_f)],
               3: [],
               4: [((3, 0), -0.04640535750777663, lin_f)]}
        intervention_variable = '1'
        intervention_value = -2.1919160604476926
        ts_old = pd.read_csv(checkpoint_path + '/TestGetIntervention_ValueError.dat')
        random_seed = 25
        n_samples = 500
        labels = ts_old.columns
        noise_type = 'without'
        assert (None, 'max_lag == 0') == data_generator(scm,
                       intervention_variable,
                       intervention_value,
                       ts_old,
                       random_seed,
                       n_samples,
                       labels,
                       noise_type)


        # given
        scm = {
            0: [((0, -1), -2.0, lin_f), ((1, 0), 5.0, lin_f)],
            1: [((0, -1), 4.0, lin_f), ((1, -1), 8.0, lin_f)],
        }
        ts = pd.DataFrame(
            [[-1.0, 0.0],
             [-2.0, 3.0]],
            columns=['0', '1'])
        random_seed = 0
        n_half_samples = 1

        # no intervention
        intervention_var = None
        intervention_value_low = None
        # when
        simulated_res, health = data_generator(
            scm=scm,
            intervention_variable=intervention_var,
            intervention_value=intervention_value_low,
            ts_old=ts,
            random_seed=random_seed,
            n_samples=n_half_samples,
            labels=ts.columns,
            noise_type='gaussian'
        )
        simulated_res = simulated_res.round(6)
        # then
        true_simulated_res = pd.DataFrame(
            [
                [79.27996, 14.46295],
            ],
            columns=['0', '1'], dtype='float32').round(6)
        assert_frame_equal(simulated_res, true_simulated_res)
        assert health == 'good'

        # with intervention
        intervention_var = '1'
        intervention_value_low = -2.0
        simulated_res, health = data_generator(
            scm=scm,
            intervention_variable=intervention_var,
            intervention_value=intervention_value_low,
            ts_old=ts,
            random_seed=random_seed,
            n_samples=n_half_samples,
            labels=ts.columns,
            noise_type='gaussian'
        )
        # then
        true_simulated_res = pd.DataFrame(
            [
                [-3.0348, -2.0],
            ],
            columns=['0', '1'], dtype='float32')
        assert_frame_equal(simulated_res.round(4), true_simulated_res.round(4))



