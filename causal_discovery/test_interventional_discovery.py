import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from causal_discovery import interventional_discovery
from causal_discovery.interventional_discovery import remove_weaker_links_of_contempt_cycles
from config import checkpoint_path


class TestInterventionalDiscovery:
    def test_get_probable_parents(self):
        # Given
        effect = '0'
        measured_labels = ['0', '1', '2', '3', '4', '5', '6', '7']
        pag_edgemarks = np.array([[['', '-->'], ['-->', '-->'], ['', '-->'], ['', '<->'], ['', '']],
                                  [['<--', '-->'], ['', '-->'], ['-->', '-->'], ['<->', ''], ['-->', '']],
                                  [['', '-->'], ['<--', '-->'], ['', '-->'], ['', ''], ['<->', '-->']],
                                  [['', 'o->'], ['<->', ''], ['', '<->'], ['', '<->'], ['', '']],
                                  [['', '<->'], ['<--', '<->'], ['<->', '-->'], ['', ''], ['', '-->']]])  # When
        probable_parents = interventional_discovery.get_probable_parents(effect, pag_edgemarks,
                                                                         measured_labels)
        # Then
        true_probable_parents = np.array([['0', '1'], ['1', '1'], ['2', '1'], ['3', '1']])
        assert np.array_equal(true_probable_parents, probable_parents)

        # 2. test
        effect = '2'
        pag_edgemarks = np.load(checkpoint_path + 'pag_edgemarks.npy', allow_pickle=True)
        probable_parents = interventional_discovery.get_probable_parents(effect, pag_edgemarks, measured_labels)
        true_probable_parents = np.array([['0', '0'], ['1', '1'], ['2', '1'], ['3', '1']])
        assert np.array_equal(true_probable_parents, probable_parents)

        # 3. test
        effect = '3'
        probable_parents = interventional_discovery.get_probable_parents(effect, pag_edgemarks,
                                                                         measured_labels)
        true_probable_parents = np.array([['3', '1']])
        assert np.array_equal(true_probable_parents, probable_parents)

    def test_remove_cause_tau_var(self):
        # Given
        cause = '0'
        tau = 1
        probable_parents = np.array([['0', '1'], ['1', '1'], ['2', '1'], ['3', '1']])
        # When
        conditioning_vars = interventional_discovery.remove_cause_tau_var(probable_parents, cause, tau)
        # Then
        true_conditioning_vars = [['1', '1'], ['2', '1'], ['3', '1']]
        assert np.array_equal(true_conditioning_vars, conditioning_vars)

    def test_get_conditioning_df(self):
        conditioning_vars = [['1', '1'], ['2', '1'], ['3', '1']]
        measured_labels = ['0', '1', '2', '3', '4', '5', '6', '7']
        df_with_intervention_on_one_cause = pd.DataFrame(
            [[9.0, 0.9490806, 0.23790693, -1.0366672, -2.7219908, -0.86635816, -0.54072285, -1.4470586],
             [8.0, 1.2169158, -0.8612138, 0.6158505, -0.7142994, 0.62477016, -2.4664948, 0.90347844],
             [9.0, -2.7442846, -0.01076746, 1.4087411, -0.66136897, 1.0595483, -2.3066196, -2.6307123],
             [8.0, -1.9110425, -2.1331735, 0.91157717, -1.5807517, 2.6822305, -1.3860753, -2.2419975],
             [9.0, -0.85678804, -2.2561557, -1.0304446, -1.3044108, 1.3641999, -0.4040094, 0.6902417]],
            columns=['0', '1', '2', '3', '4', '5', '6', '7'])
        solution = pd.DataFrame(
            [
                [np.NaN, np.NaN, np.NaN],
                [0.94908, 0.23791, -1.03667],
                [1.21692, -0.86121, 0.61585],
                [-2.74428, -0.01077, 1.40874],
                [-1.91104, -2.13317, 0.91158]
            ], columns=['1_1', '2_1', '3_1']).round(5)
        res = interventional_discovery.get_conditioning_df(conditioning_vars, df_with_intervention_on_one_cause,
                                                           measured_labels).round(5)
        assert_frame_equal(res, solution)

    def test_align_cause_effect_due_to_lag(self):
        # Given
        cause_and_effect_tau_shifted = pd.DataFrame([[5.0, -1.0], [6.0, 0.0], [7.0, 1.0], [8.0, 2.0]],
                                                    columns=['cause', 'effect'])
        # When
        aligned_cause_and_effect_tau_shifted = interventional_discovery.align_cause_effect_due_to_lag(
            cause_and_effect_tau_shifted, 1)
        # Then
        true_aligned_cause_and_effect_tau_shifted = pd.DataFrame([[np.NaN, -1.0], [5.0, 0.0], [6.0, 1.0], [7.0, 2.0]],
                                                                 columns=['cause', 'effect'])
        assert aligned_cause_and_effect_tau_shifted.equals(true_aligned_cause_and_effect_tau_shifted)

        # for tau=0
        # When
        aligned_cause_and_effect_tau_shifted = interventional_discovery.align_cause_effect_due_to_lag(
            cause_and_effect_tau_shifted, 0)
        # Then
        true_aligned_cause_and_effect_tau_shifted = cause_and_effect_tau_shifted
        assert aligned_cause_and_effect_tau_shifted.equals(true_aligned_cause_and_effect_tau_shifted)

    def test_get_independencies_from_interv_data(self):
        # Given
        df = pd.read_pickle(checkpoint_path + 'df.pkl')
        was_intervened = pd.read_pickle(checkpoint_path + 'was_intervened.pkl')
        interv_alpha = 0.7 / 3
        n_ini_obs = 500
        pag_edgemarks = np.load(checkpoint_path + 'pag_edgemarks.npy', allow_pickle=True)
        measured_labels = ['0', '2', '3', '4', '5']
        # When
        indepdendencies, dependencies = interventional_discovery.get_independencies_from_interv_data(df, was_intervened,
                                                                                                     interv_alpha,
                                                                                                     n_ini_obs,
                                                                                                     pag_edgemarks,
                                                                                                     measured_labels)
        # Then
        true_indepdendencies = [('0', '3', 1, 0.404), ('4', '3', 0, 0.817), ('4', '3', 1, 0.993), ('5', '3', 0, 0.664),
                                ('5', '3', 1, 0.87)]
        assert np.array_equal(true_indepdendencies, indepdendencies)

    def test_remove_weaker_links_of_contempt_cycles(self):
        dependencies_from_interv_data = [
            ('0', '2', 0, 0.118),
            ('3', '2', 0, 0.145),
            ('0', '3', 1, 0.009),
            ('2', '3', 0, 0.012),
            ('5', '3', 0, 0.001)
        ]
        dependencies_from_interv_data = remove_weaker_links_of_contempt_cycles(dependencies_from_interv_data)
        assert dependencies_from_interv_data == [
            ('0', '2', 0, 0.118),
            ('0', '3', 1, 0.009),
            ('2', '3', 0, 0.012),
            ('5', '3', 0, 0.001)
        ]

        dependencies_from_interv_data = [
            ('3', '0', 1, 0.1), ('0', '3', 1, 0.2),
            ('3', '1', 0, 0.1), ('1', '3', 0, 0.2),
            ('4', '0', 1, 0.1), ('0', '4', 1, 0.2),
            ('4', '1', 0, 0.1), ('1', '4', 0, 0.2),
        ]
        dependencies_from_interv_data = remove_weaker_links_of_contempt_cycles(dependencies_from_interv_data)
        assert dependencies_from_interv_data == [
            ('3', '0', 1, 0.1), ('0', '3', 1, 0.2),
            ('3', '1', 0, 0.1),
            ('4', '0', 1, 0.1), ('0', '4', 1, 0.2),
            ('4', '1', 0, 0.1)
        ]


