import math

import numpy as np


def ensure_0_in_measured_labels(measured_labels):
    if 0 not in measured_labels:
        # remove last element of measured_labels
        measured_labels = measured_labels[:-1]
        # add 0 to measured_labels
        measured_labels.append(0)
        measured_labels = np.sort(measured_labels).tolist()
    return measured_labels


def get_measured_labels(n_vars_all, random_state, frac_latents):
    all_labels_ints = range(n_vars_all)
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

    unmeasured_labels_strs = [str(x) for x in unmeasured_labels_ints]


    # measured_labels to strings
    measured_labels = [str(x) for x in measured_labels]

    """ key value map of label to index """
    measured_label_as_idx = {label: idx for idx, label in enumerate(measured_labels)}

    return measured_labels, measured_label_as_idx, unmeasured_labels_strs