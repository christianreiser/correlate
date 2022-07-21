import unittest

import pandas as pd
from statsmodels.compat.pandas import assert_frame_equal

from data_generation import data_generator
from intervention_proposal.get_intervention import lin_f


class TestGetIntervention(unittest.TestCase):
    def test_data_generator(self):
        # given
        scm = {
            0: [((0, -1), -2.0, lin_f), ((1, 0), 5.0, lin_f)],
            1: [((0, -1), 4.0, lin_f), ((1, -1), 8.0, lin_f)],
        }
        intervention_var = None
        intervention_value_low = None
        ts = pd.DataFrame(
            [[-1.0, 0.0],
             [-2.0, 3.0]],
            columns=['0', '1'])
        random_seed = 0
        n_half_samples = 1
        # when
        simulated_res = data_generator(
            scm=scm,
            intervention_variable=intervention_var,
            intervention_value=intervention_value_low,
            ts_old=ts,
            random_seed=random_seed,
            n_samples=n_half_samples,
            labels=ts.columns
        ).round(6)
        # then
        true_simulated_res = pd.DataFrame(
            [
                [79.27996, 14.46295],
            ],
            columns=['0', '1'], dtype='float32').round(6)
        assert_frame_equal(simulated_res, true_simulated_res)


        intervention_var = '1'
        intervention_value_low = -2.0
        simulated_res = data_generator(
            scm=scm,
            intervention_variable=intervention_var,
            intervention_value=intervention_value_low,
            ts_old=ts,
            random_seed=random_seed,
            n_samples=n_half_samples,
            labels=ts.columns
        ).round(4)
        # then
        true_simulated_res = pd.DataFrame(
            [
                [-3.0348, -2.0],
            ],
            columns=['0', '1'], dtype='float32').round(4)
        assert_frame_equal(simulated_res, true_simulated_res)


if __name__ == '__main__':
    unittest.main()
