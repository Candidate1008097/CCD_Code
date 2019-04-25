import numpy as np
import matplotlib.pyplot as plt
from solve import solve, current

force_calculation = True

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
x_max = 6

# model functions
E = lambda t: -10

# solve
tts, xxs, U = solve(alpha, k_0, E, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max, True, 'constant', force_calculation)
ts, I = current(tts, xxs, U)

# plot
plt.plot(ts, I)
plt.xlabel(r'$t$ (time)')
plt.xticks(np.linspace(0, 20, 11))
plt.ylabel(r'$I$ (current)')
plt.grid(True)

# save plots
# plt.savefig('figures/implicit_euler_solve.pgf', bbox_inches='tight')
# plt.savefig('figures/implicit_euler_solve.png', bbox_inches='tight')

plt.show()
