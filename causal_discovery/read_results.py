import pickle

import numpy as np

from config import checkpoint_path


# load
with open(checkpoint_path + str(0) + 'regret_list_over_simulation_study.pickle', 'rb') as f:
    regret_list_over_simulation_study, simulation_study = pickle.load(f)

for regret_loss, setting in zip(regret_list_over_simulation_study, simulation_study):
    regret_loss = np.array(regret_loss)
    regret = regret_loss[:,0]
    loss = regret_loss[:,1]
    mean_regret = np.mean(regret)
    mean_loss = np.mean(loss)
    print('\nsetting:',setting,
          '\nmean_regret:',mean_regret,
          '\nmean_loss', mean_loss,
          '\nregret',regret,
          '\nloss',loss,'\n')

print()