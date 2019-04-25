import numpy as np
import matplotlib.pyplot as plt
from solve import solve, current

force_calculation = True

# model parameters
alphas = np.linspace(0, 1, 6)
k_0s = 2.0**np.array([-3, -1, 1, 3])
E0 = 0
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
for n, k_0 in enumerate(k_0s):
    plt.subplot(2, 2, n + 1)

    for alpha in alphas:
        tts, xxs, U = solve(alpha, k_0, E, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max,
                            True, 'vary/alpha', force_calculation)
        ts, I = current(tts, xxs, U)

        # plot
        plt.plot(ts, I, label=r'$\alpha={:.1f}$'.format(alpha), linewidth=1)

    if int(k_0) == k_0:
        plt.title(r'$k_0={}$, $E_0={}$'.format(int(k_0), E0))
    else:
        plt.title(r'$k_0={}$, $E_0={}$'.format(k_0, E0))
    plt.xlabel(r'$t$ (time)')
    plt.ylabel(r'$I$ (current)')
    plt.grid(True)

# save plots
plt.savefig('figures/vary_alpha.pgf')
plt.savefig('figures/vary_alpha.png')

plt.show()
