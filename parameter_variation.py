from solve import solve, current
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

force_calculation = False
do_main_loop = False

# vary the parameters to see what happens
n_alpha = 20
n_E0 = 21
alphas = np.linspace(0, 1, n_alpha+1, False)[1:]  # remove the 0 from the beginning
E0s = np.linspace(-5, 5, n_E0+1, False)[1:]  # again, remove the first one
k_0s = 10.0**np.arange(-2, 4)

# constants
E_start = -10
E = lambda t: t + E_start  # no AC variation, just the DC sweep

# boundary conditions
u_0 = lambda x: np.full_like(x, 1)
u_Inf = 1

# numerical parameters
delta_t = 0.1
delta_x = 0.1
t_max = 20
x_max = 6

peak_currents = np.zeros((k_0s.size, n_alpha, n_E0))
peak_times = np.zeros_like(peak_currents)

# loop through and solve
if do_main_loop:
    for i, k_0 in enumerate(k_0s):
        for j, alpha in enumerate(alphas):
            for k, E0 in enumerate(E0s):
                # let us know it's churning away
                print('k_0 = {}, alpha = {}, E0 = {}'.format(k_0, alpha, E0))

                # solve the problem and find the current
                tts, xxs, U = solve(alpha, k_0, E, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max,
                                    True, 'k_0={}/alpha={}/E0={}'.format(k_0, alpha, E0), force_calculation)
                ts, I = current(tts, xxs, U)

                # save peak
                max_index = np.argmax(I)  # index of peak current value, used to find the time at which it occured
                peak_currents[i, j, k] = I[max_index]
                peak_times[i, j, k] = ts[max_index]

    # save data so we can reuse it
    if not os.path.exists('data/parameter_variation'):
        os.makedirs('data/parameter_variation')
    np.save('data/parameter_variation/peak_currents', peak_currents)
    np.save('data/parameter_variation/peak_times', peak_times)
else:
    peak_currents = np.load('data/parameter_variation/peak_currents.npy')
    peak_times = np.load('data/parameter_variation/peak_times.npy')

# how k_0 varies alpha-dependence
plt.figure('k_0 varying alpha-dependence')
# use the values at E0 = 0
zero_index = int(np.argmin(np.abs(E0s)))
for i, k_0 in enumerate(k_0s):
    plt.plot(alphas, peak_currents[i, :, zero_index], label=r'$k_0={}$'.format(k_0))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.xlabel(r'$\alpha$')
plt.yticks(np.arange(0, 0.6, 0.1))
plt.ylabel(r'Peak current $I_{max}$')
plt.grid()
plt.legend()
plt.title(r'Effect of $\alpha$ being varied by $k_0$ ($E^0 = {:.2f}$)'.format(E0s[zero_index]))

# how E0 changes peak time
plt.figure('E0 varying peak time')
# use the values close to alpha = 0.5
half_index = np.argmin(np.abs(alphas - 0.5))
for i, k_0 in enumerate(k_0s):
    plt.plot(E0s, peak_times[i, half_index, :], label=r'$k_0={}$'.format(k_0))
plt.xlabel(r'$E^0$')
plt.ylabel(r'Time $t$ of peak current $I_{max}$')
plt.grid()
plt.legend()
plt.title(r'Effect of $E^0$ on time of peak current ($\alpha = {:.2f}$)'.format(alphas[half_index]))

plt.show()
