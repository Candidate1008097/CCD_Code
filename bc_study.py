import numpy as np
import matplotlib.pyplot as plt
from solve import solve, current

force_calculation = False

# model parameters
alpha = 0.5
k_0 = 35
E0 = 0
E_start = -10

# boundary conditions
u_0 = lambda x: np.full_like(x, 1)  # u(x, t=0)
u_Inf = 1  # u(x -> Inf, t>=0), we will be prescribing this at x_max instead

# numerical parameters
delta_t = 0.1
delta_x = 0.1
t_max = 20
x_max = 20

# model functions
E = lambda t: t + E_start

timesteps = [0, 5, 10, 15, 20]

# solve
tts, xxs, U = solve(alpha, k_0, E, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max,
                    True, 'xmax/xmax={}'.format(x_max), force_calculation, verbose=True)
ts, I = current(tts, xxs, U)

# plot
for timestep in timesteps:
    # find index
    i = np.argmin(np.abs(ts - timestep))

    # plot
    plt.plot(xxs[i, :], U[i, :], label=r'$t={}$'.format(timestep))

plt.legend()
plt.grid()
plt.xlabel(r'Distance from electrode $x$')
plt.ylabel(r'Concentration $a$')

# save plots
plt.savefig('figures/conc_xmax_{}.png'.format(x_max))
plt.savefig('figures/conc_xmax_{}.pgf'.format(x_max))

plt.show()
