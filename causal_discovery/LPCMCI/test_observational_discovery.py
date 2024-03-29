import pickle

from matplotlib import pyplot as plt
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr

from causal_discovery.LPCMCI.lpcmci import LPCMCI

from config import checkpoint_path


class TestLPCMCI:

    def test_orient_with_interv_data(self):
        interv_independencies = [(0, 3, 0), (0, 3, 1), (1, 3, 0), (1, 3, 1), (2, 3, 0), (2, 3, 1)]
        interv_dependencies = [(4, 2, 0)]
        self.graph_dict = {
            0: {(1, 0): 'o?o', (2, 0): 'o?o', (3, 0): 'o?o', (4, 0): 'o?o', (0, -1): 'oL>', (1, -1): 'oL>',(2, -1): 'oL>', (3, -1): 'oL>', (4, -1): 'oL>'},
            1: {(0, 0): 'o?o', (2, 0): 'o?o', (3, 0): 'o?o', (4, 0): 'o?o', (0, -1): 'oL>', (1, -1): 'oL>',(2, -1): 'oL>', (3, -1): 'oL>', (4, -1): 'oL>'},
            2: {(0, 0): 'o?o', (1, 0): 'o?o', (3, 0): 'o?o', (4, 0): 'o?o', (0, -1): 'oL>', (1, -1): 'oL>',(2, -1): 'oL>', (3, -1): 'oL>', (4, -1): 'oL>'},
            3: {(0, 0): 'o?o', (1, 0): 'o?o', (2, 0): 'o?o', (4, 0): 'o?o', (0, -1): 'oL>', (1, -1): 'oL>',(2, -1): 'oL>', (3, -1): 'oL>', (4, -1): 'oL>'},
            4: {(0, 0): 'o?o', (1, 0): 'o?o', (2, 0): 'o?o', (3, 0): 'o?o', (0, -1): 'oL>', (1, -1): 'oL>',(2, -1): 'oL>', (3, -1): 'oL>', (4, -1): 'oL>'}
        }

        # independencies
        if interv_independencies is not None and len(interv_independencies) > 0:
            for independency in interv_independencies:
                eff = (independency[0], independency[2])
                cause = (independency[1], 0)
                (var_cause, lag_cause) = cause
                (var_eff, lag_eff) = eff
                # if self.graph_dict[var_eff][(var_cause, lag_cause - lag_eff)] != "":
                if self.graph_dict[var_eff][(var_cause, lag_cause - lag_eff)][0] in ["o"]:
                    self.graph_dict[var_eff][(var_cause, lag_cause - lag_eff)] = "<" + str(
                        self.graph_dict[var_eff][(var_cause, lag_cause - lag_eff)][1:])
                    # If A and B are contemporaneous, also the link from B to A is written as the reverse
                    if lag_eff == 0:
                        self.graph_dict[var_cause][(var_eff, 0)] = str(
                            self.graph_dict[var_cause][(var_eff, 0)][:2]) + ">"
                else:
                    raise ValueError("orient with_interv_data: unexpected edgemark. expected o but is:",
                               self.graph_dict[var_eff][(var_cause, lag_cause - lag_eff)][0])

        # dependencies
        if interv_dependencies is not None and len(interv_dependencies) > 0:
            for dependency in interv_dependencies:
                eff = (dependency[0], dependency[2])
                cause = (dependency[1], 0)
                (var_cause, lag_cause) = cause
                (var_eff, lag_eff) = eff
                # if self.graph_dict[var_eff][(var_cause, lag_cause - lag_eff)] != "":
                if self.graph_dict[var_eff][(var_cause, lag_cause - lag_eff)][0] in ["o"] and \
                        self.graph_dict[var_eff][(var_cause, lag_cause - lag_eff)][2] in ["o", ">"]:
                    self.graph_dict[var_eff][(var_cause, lag_cause - lag_eff)] = "-" + str(
                        self.graph_dict[var_eff][(var_cause, lag_cause - lag_eff)][1] + ">")
                    # If A and B are contemporaneous, also the link from B to A is written as the reverse
                    if lag_eff == 0:
                        self.graph_dict[var_cause][(var_eff, 0)] = "<"+ str(
                            self.graph_dict[var_cause][(var_eff, 0)][1]) + "-"
                else:
                    raise ValueError("orient with_interv_data: unexpected edgemark. expected o but is:",
                               self.graph_dict[var_eff][(var_cause, lag_cause - lag_eff)][0])

        solution_graph_dict = {
            0: {(1, 0): 'o?o', (2, 0): 'o?o', (3, 0): '<?o', (4, 0): 'o?o', (0, -1): 'oL>', (1, -1): 'oL>',(2, -1): 'oL>', (3, -1): '<L>', (4, -1): 'oL>'},
            1: {(0, 0): 'o?o', (2, 0): 'o?o', (3, 0): '<?o', (4, 0): 'o?o', (0, -1): 'oL>', (1, -1): 'oL>',(2, -1): 'oL>', (3, -1): '<L>', (4, -1): 'oL>'},
            2: {(0, 0): 'o?o', (1, 0): 'o?o', (3, 0): '<?o', (4, 0): '<?-', (0, -1): 'oL>', (1, -1): 'oL>',(2, -1): 'oL>', (3, -1): '<L>', (4, -1): 'oL>'},
            3: {(0, 0): 'o?>', (1, 0): 'o?>', (2, 0): 'o?>', (4, 0): 'o?o', (0, -1): 'oL>', (1, -1): 'oL>',(2, -1): 'oL>', (3, -1): 'oL>', (4, -1): 'oL>'},
            4: {(0, 0): 'o?o', (1, 0): 'o?o', (2, 0): '-?>', (3, 0): 'o?o', (0, -1): 'oL>', (1, -1): 'oL>',(2, -1): 'oL>', (3, -1): 'oL>', (4, -1): 'oL>'}}
        assert self.graph_dict == solution_graph_dict

    def test_run_lpcmci(self):
        # load
        filename = checkpoint_path + 'test_run_lpcmci.pkl'
        with open(filename, 'rb') as f:
            df, _, _, _ = pickle.load(f)
            pc_alpha = 0.05
            tau_max = 1
        # (effect, cause, tau, p-val)
        external_independencies = [(0, 3, 0), (0, 3, 1), (1, 3, 0), (1, 3, 1), (2, 3, 0), (2, 3, 1)]
        external_dependencies = [ (4, 2, 1)]

        # run lpcmci
        lpcmci = LPCMCI(
            dataframe=df,
            cond_ind_test=ParCorr(
                significance='analytic',
                recycle_residuals=True))

        lpcmci.run_lpcmci(
            external_independencies=external_independencies,
            external_dependencies=external_dependencies,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            max_p_non_ancestral=2,  # todo 3
            n_preliminary_iterations=1,  # todo 4
            prelim_only=False,
            verbosity=0)
        graph = lpcmci.graph

        # tp.plot_graph(
        #     val_matrix=lpcmci.val_min_matrix,
        #     link_matrix=graph,
        #     var_names=["0", "2", "3", "4", "5"],
        #     link_colorbar_label='current LPCMCI estimate. day',
        #     node_colorbar_label='auto-MCI',
        #     figsize=(10, 6),
        # )
        # plt.show()

        for exi in external_independencies:
            exi = list(exi)
            forward_arrow = graph[exi[1], exi[0], exi[2]]
            assert forward_arrow == "" or forward_arrow[0] == "<"
            # symmetric for contemporaneous links
            if exi[2] == 0:
                backward_arrow = graph[exi[0], exi[1], exi[2]]
                assert backward_arrow == "" or backward_arrow[2] == ">"

        for exi in external_dependencies:
            exi = list(exi)
            forward_arrow = graph[exi[1], exi[0], exi[2]]
            assert forward_arrow == "" or forward_arrow[0] == "-"
            assert forward_arrow == "" or forward_arrow[2] == ">"
            # symmetric for contemporaneous links
            if exi[2] == 0:
                backward_arrow = graph[exi[0], exi[1], exi[2]]
                assert backward_arrow == "" or backward_arrow[2] == "-"
                assert backward_arrow == "" or backward_arrow[0] == "<"
