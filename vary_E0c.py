import numpy as np
import matplotlib.pyplot as plt
from solve_ece import solve_ece, current_ece

force_calculation = False

# model parameters
alpha = 0.5
ka_0 = 35
kc_0 = 35
r = 1
E0a = 0
E0cs = np.arange(-4, 11, 2)
E_start = -10

# numerical parameters
delta_t = 0.1
delta_x = 0.1
t_max = 20
x_max = 6

# model functions
E = lambda t: t + E_start

# solve for all alphas
for E0c in E0cs:
    tts, xxs, A, B, C = solve_ece(alpha, ka_0, kc_0, r, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                        True, 'vary/E0c/E0c={}'.format(E0c), force_calculation)
    ts, I = current_ece(tts, xxs, A, C)

    # plot
    plt.plot(ts, I, label=r'$E_0^c={}$'.format(E0c))

plt.legend()
plt.xlabel(r'$t$ (time)')
plt.xticks(np.linspace(0, 20, 11))
plt.ylabel(r'$I$ (current)')
plt.grid(True)

# save plots
plt.savefig('figures/vary_E0c.pgf')
plt.savefig('figures/vary_E0c.png')

plt.show()
