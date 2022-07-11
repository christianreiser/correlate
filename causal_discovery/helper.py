import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt

def parabola(x, a, b, c):
    x = np.array(x)
    return a * x ** 2 + b * x + c


def arg_closest(lst, x):
    lst = np.subtract(lst, x)
    return np.where(lst == min(lst, key=abs))[0][0]


# def reduce_tau_max(correlations):
#     # 3d-> 2D via reshape, 2D->1D via amax, abs
#     abs_max_corr_coeff = np.absolute(np.amax(correlations.reshape(correlations.shape[0] ** 2, -1), axis=0))
#
#     abs_max_corr_coeff = np.delete(abs_max_corr_coeff, 0)  # remove idx 0. idk what it was for
#     time_lag = list(range(0, len(abs_max_corr_coeff)))  # array of time lags
#     parabola_params, _ = scipy.optimize.curve_fit(parabola, time_lag, abs_max_corr_coeff)  # parabola_params
#     y_parabola = parabola(time_lag, *parabola_params)  # y values of fitted parabola
#     parabola_first_half = y_parabola[:np.argmin(y_parabola)]  # keep only part of parabola which is before argmin
#     tau_max = arg_closest(parabola_first_half, corr_threshold)
#     print('reduced tau_max=', tau_max)
#
#     # plotting
#     plt.plot(abs_max_corr_coeff, label='max correlation coefficient', color='black')
#     plt.plot(time_lag, y_parabola, label='quadratic fit', color='blue')
#     plt.fill_between([0, len(abs_max_corr_coeff)], 0, corr_threshold,
#                      facecolor='red', alpha=0.3, label='below corr threshold')
#     plt.axvline(tau_max, 0, 30, label='tau_max', color='red')
#     plt.title('Computation of tau_max=' + str(tau_max))
#     plt.ylabel('max correlation coefficient')
#     plt.ylabel('time lag')
#     plt.xlim([0, len(abs_max_corr_coeff)])
#     plt.ylim([0, max(abs_max_corr_coeff)])
#     plt.legend(loc='best')
#     plt.show()
#     return tau_max
