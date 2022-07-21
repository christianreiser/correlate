import pickle
import unittest

import numpy as np

from config import checkpoint_path
from intervention_proposal.get_intervention import find_optimistic_intervention, \
    drop_redundant_information_due_to_symmetry, get_ambiguous_graph_locations, create_all_graph_combinations, \
    graph_to_scm, lin_f


class TestGetIntervention(unittest.TestCase):
    def test_drop_redundant_information_due_to_symmetry(self):
        # Given
        original_graph = np.array([[['', '0->'], ['-->', '-->']], [['<--', '<->'], ['', '-->']]])
        # When
        modified_graph = drop_redundant_information_due_to_symmetry(original_graph)
        # Then
        true_graph = np.array([[['', '0->'], ['', '-->']], [['<--', '<->'], ['', '-->']]])
        self.assertTrue(np.array_equal(true_graph, modified_graph))

    def test_get_ambiguous_graph_locations(self):
        # Given
        my_graph = np.array([[['', 'o->'], ['', '-->']], [['x-x', '<->'], ['', '-->']]])
        # When
        ambiguous_locations = get_ambiguous_graph_locations(my_graph)
        # Then
        true_ambiguous_locations = [
            [0, 0, 1, 'o->', ["-->", "<->"]],
            [1, 0, 0, 'x-x', ["-->", "<->", "<--"]],
        ]
        self.assertTrue(np.array_equal(true_ambiguous_locations, ambiguous_locations))

        # 2. test empty
        # Given
        my_graph = np.array([[['', '-->'], ['', '-->']], [['<--', '<->'], ['', '-->']]])
        # When
        ambiguous_locations = get_ambiguous_graph_locations(my_graph)
        # Then
        true_ambiguous_locations = []
        self.assertTrue(np.array_equal(true_ambiguous_locations, ambiguous_locations))

    def test_create_all_graph_combinations(self):
        # given
        my_graph = np.array([[['', 'o->'], ['', '-->']], [['x->', '<->'], ['', '-->']]])
        ambiguous_locations = [
            [0, 0, 1, 'o->', ["-->", "<->"]],
            [1, 0, 0, 'x->', ["-->", "<->"]],
        ]
        # when
        all_graph_combinations = create_all_graph_combinations(my_graph, ambiguous_locations)
        # then
        true_all_graph_combinations = [
            np.array([[['', '-->'], ['', '-->']], [['-->', '<->'], ['', '-->']]]),
            np.array([[['', '-->'], ['', '-->']], [['<->', '<->'], ['', '-->']]]),
            np.array([[['', '<->'], ['', '-->']], [['-->', '<->'], ['', '-->']]]),
            np.array([[['', '<->'], ['', '-->']], [['<->', '<->'], ['', '-->']]]),
        ]
        self.assertTrue(np.array_equal(true_all_graph_combinations, all_graph_combinations))

    def test_graph_to_scm(self):
        # Given
        my_graph = np.array([[['', '-->'], ['', '-->']], [['-->', '<->'], ['', '-->']]])
        val = np.array([[[0.0, 2.0], [0.0, 4.0]], [[5.0, 6.0], [0.0, 8.0]]])
        # When
        scm = graph_to_scm(my_graph, val)
        # Then
        true_scm = {
            0: [((0, -1), 2.0, lin_f), ((1, 0), 5.0, lin_f)],
            1: [((0, -1), 4.0, lin_f), ((1, -1), 8.0, lin_f)],
        }
        self.assertTrue(np.array_equal(true_scm, scm))

    def test_find_optimistic_intervention(self):
        # Given
        # load from file
        with open(checkpoint_path + '{}.pkl'.format('true_scm'), 'rb') as f:
            my_graph, val, var_names, ts, unintervenable_vars, random_seed, old_intervention, label, external_independencies = pickle.load(
                f)
        # When
        intervention = find_optimistic_intervention(my_graph, val, var_names, ts, unintervenable_vars, random_seed,
                                                    old_intervention, label, external_independencies)
        # Then
        true_intervention = {'0': 1, '1': 1, '2': 1, '3': 1}


if __name__ == '__main__':
    unittest.main()
