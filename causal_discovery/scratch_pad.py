# load target_ans_per_graph_dict from file via pickle
import pickle

with open('target_eq_chr.pkl', 'rb') as f:
    target_eq_chr = pickle.load(f)

print('')