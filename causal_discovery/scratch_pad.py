from itertools import product

auto_first = False
# auto_first = True

N=7
tau_max=1


link_list = [product(range(N), range(N), range(-lag, -lag + 1)) for lag in
             range(0, tau_max + 1)]

# Run through all elements of link_list. Each element of link_list specifies ordered pairs of variables whose connecting edges are then subjected to conditional independence tests
link_idx = -1
for links in link_list:
    link_idx += 1

    pair_idx = -1
    # Iterate through all edges specified by links. Note that since the variables paris are ordered, (A, B) and (B, A) are seen as different pairs.
    for pair in links:
        pair_idx += 1
        print('link_idx', link_idx, 'pair_idx', pair_idx, 'pair', pair)

        # Decode the elements of links into pairs of variables (X, Y)
        X = (pair[0], pair[2])
        Y = (pair[1], 0)

        ######################################################################################################
        ### Exclusion of links ###############################################################################

        # Exclude the current link if ...
        # ... X = Y
        if X[1] == 0 and X[0] == Y[0]:
            continue
        # ... X > Y
        # if self._is_smaller(Y, X):
        #     continue

print()