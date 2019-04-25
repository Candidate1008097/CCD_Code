import math
import numpy as np
import matplotlib.pyplot as plt
from solve import solve, current

# calculate everything from scratch, or read from file if it exists?
force_calculation = False

# different frequencies and amplitudes to test
omegas = np.array([0.5, 1, 2, 4, 8, 16, 32])*math.pi
delta_Es = np.arange(0.05, 1.05, 0.05)#/38.92  # ac amplitude

# unchanged parameters
alpha = 0.5
k_0 = 35
E0 = 0
E_start = -10


def gen_E(delta_E, omega):
    def E(t):
        return t + E_start + delta_E*math.sin(omega*t)

    return E


u_0 = lambda x: 1
u_Inf = 1

# numerical parameters
delta_x = 0.1
x_max = 6
t_max = 20

# final result meshes
peak_currents = np.zeros((delta_Es.size, omegas.size))

# iterate through different frequencies
for O, omega in enumerate(omegas):
    # change delta t each time so there are 100 steps per ac period
    delta_t = (2*math.pi/omega)/100

    # solve for no ac component
    tts, xxs, U_0 = solve(alpha, k_0, gen_E(0, omega), E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max, True,
                          'omega={}pi/DC_only'.format(omega/math.pi), force_calculation)

    # find base current
    _, I_0 = current(tts, xxs, U_0)

    # print something, so we know it's doing work
    print('Omega: ', omega/math.pi, '*pi, delta E: ', 0, sep='')

    # then calculate the current with the ac wave at different amplitudes
    for D, delta_E in enumerate(delta_Es):
        # solve
        tts, xxs, U = solve(alpha, k_0, gen_E(delta_E, omega), E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max, True,
                            'omega={}pi/delta_E={}'.format(omega/math.pi, delta_E), force_calculation)

        # find ac current
        _, I = current(tts, xxs, U)

        # calculate peak of ac component alone
        peak = np.max(np.abs(I - I_0))

        # store peak
        peak_currents[D, O] = peak

        print('Omega: ', omega/math.pi, '*pi, delta E: ', delta_E, sep='')

# plot results
plt.figure(figsize=(4.768, 4.768))
plt.plot(delta_Es, peak_currents, 'k', label='Simulated', linewidth=1)
plt.xlim(0, 1.3)

# linear extrapolations
lin_dE = np.array([delta_Es[0], 1])
lin_I = np.zeros((2, omegas.size))

for i in range(omegas.size):
    # text label
    plt.text(delta_Es[-1] + 0.03, peak_currents[-1, i] - 0.03, r'$\Omega={}\pi$'.format(omegas[i]/math.pi),
             horizontalalignment='left')

    # linear extrapolation
    gradient = (peak_currents[1, i] - peak_currents[0, i]) / (delta_Es[1] - delta_Es[0])
    y_intercept = peak_currents[0, i] - gradient * delta_Es[0]
    lin_I[:, i] = y_intercept + gradient * lin_dE
plt.plot(lin_dE, lin_I, 'k--', label='True linearity', linewidth=1)
plt.xlabel(r'Amplitude $\Delta E$ of AC variation')
plt.ylabel(r'Amplitude peak $I_p$ of the AC component')
plt.legend(plt.gca().get_legend_handles_labels()[0][omegas.size - 1:omegas.size + 1],
           plt.gca().get_legend_handles_labels()[1][omegas.size - 1:omegas.size + 1])

# save
plt.savefig('figures/ac_variation.pgf')
plt.savefig('figures/ac_variation.png')

# show
plt.show()
