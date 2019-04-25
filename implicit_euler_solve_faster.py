import numpy as np
import matplotlib.pyplot as plt
from solve_faster import solve, current

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
E = lambda t: t + E_start

# solve
tts, xxs, U = solve(alpha, k_0, E, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max, True, 'check', force_calculation)
ts, I = current(tts, xxs, U)

# plot
plt.plot(ts, I, 'k', label='Numerical simulation')

# load experimental data
data_path = 'data/bardst.d'
data_file = open(data_path, 'r')
data_t = []
data_I = []
data_mult_constant = -38.92/1000
data_start_constant = -10
for data_line in data_file:
    t_I = data_line.split()
    t, I = float(t_I[0]), float(t_I[1])
    data_t.append(t*data_mult_constant - data_start_constant)
    data_I.append(I)

# plot experimental data
plt.plot(data_t, data_I, '.k', label='Experimental data')
plt.legend()
plt.xlabel(r'$t$ (time)')
plt.xticks(np.linspace(0, 20, 11))
plt.ylabel(r'$I$ (current)')
plt.yticks(np.linspace(0, 0.45, 10))
plt.grid(True)

# save plots
# plt.savefig('figures/implicit_euler_solve.pgf', bbox_inches='tight')
# plt.savefig('figures/implicit_euler_solve.png', bbox_inches='tight')

plt.show()
