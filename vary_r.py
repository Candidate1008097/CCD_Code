import numpy as np
import matplotlib.pyplot as plt
from solve_ece import solve_ece, current_ece

force_calculation = False

# model parameters
alpha = 0.5
ka_0 = 35
kc_0 = 35
rs = 2.0**np.arange(0, 10)
E0a = 0
E0c = 0
E_start = -10

# numerical parameters
delta_t = 0.1
delta_x = 0.1
t_max = 20
x_max = 6

# model functions
E = lambda t: t + E_start

# solve for all alphas
for r in rs:
    tts, xxs, A, B, C = solve_ece(alpha, ka_0, kc_0, r, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                        True, 'vary/r/r={}'.format(r), force_calculation)
    ts, I = current_ece(tts, xxs, A, C)

    # plot
    if int(r) == r:
        plt.plot(ts, I, label=r'$r={}$'.format(int(r)))
    else:
        plt.plot(ts, I, label=r'$r={}$'.format(r))

plt.legend()
plt.xlabel(r'$t$ (time)')
plt.xticks(np.linspace(0, 20, 11))
plt.ylabel(r'$I$ (current)')
plt.grid(True)

# save plots
plt.savefig('figures/vary_r.pgf')
plt.savefig('figures/vary_r.png')

plt.show()
