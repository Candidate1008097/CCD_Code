import numpy as np
import matplotlib.pyplot as plt
from solve import solve, current

force_calculation = True

# model parameters
alpha = 0.5
k_0 = 35
E0s = np.arange(-6, 7, 2)
E_start = -10

# boundary conditions
u_0 = lambda x: np.full_like(x, 1)  # u(x, t=0)
u_Inf = 1  # u(x -> Inf, t>=0), we will be prescribing this at x_max instead

# numerical parameters
delta_t = 0.1
delta_x = 0.1
t_max = 20
x_max = 6

# model functions
E = lambda t: t + E_start

# solve for all alphas
for E0 in E0s:
    tts, xxs, U = solve(alpha, k_0, E, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max,
                        True, 'vary/E0', force_calculation)
    ts, I = current(tts, xxs, U)

    # plot
    if int(E0) == E0:
        plt.plot(ts, I, label=r'$E_0={}$'.format(int(E0)))
    else:
        plt.plot(ts, I, label=r'$E_0={}$'.format(E0))

plt.legend(loc='center left')
plt.title(r'$\alpha={}$, $k_0={}$'.format(alpha, k_0))
plt.xlabel(r'$t$ (time)')
plt.xticks(np.linspace(0, 20, 11))
plt.ylabel(r'$I$ (current)')
plt.yticks(np.linspace(0, 0.45, 10))
plt.grid(True)

# save plots
plt.savefig('figures/vary_E0.pgf')
plt.savefig('figures/vary_E0.png')

plt.show()
