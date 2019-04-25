from solve import solve, current
from solve_ece import solve_ece, current_ece
import numpy as np
import matplotlib.pyplot as plt

force_calculation = False

# model parameters
alpha = 0.5
ka_0 = 35
kc_0 = 35
k = 1
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

# solve
tts, xxs, A_eul = solve(alpha, ka_0, E, E0a, lambda x: 1, 1, delta_t, delta_x, t_max, x_max,
                              True, 'standard', force_calculation)
tts_ece, xxs_ece, A, B, C = solve_ece(alpha, ka_0, kc_0, k, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                              True, 'ece', force_calculation)
ts_eul, I_eul = current(tts, xxs, A_eul)
ts_ece, I_ece = current_ece(tts_ece, xxs_ece, A, C)

# plot
plt.plot(ts_ece, I_ece, label=r'$ECE$ reaction')
plt.plot(ts_eul, I_eul, '--', label='Standard reaction')
plt.legend()
plt.xlabel(r'$t$ (time)')
plt.xticks(np.linspace(0, 20, 11))
plt.ylabel(r'$I$ (current)')
# plt.yticks(np.linspace(0, 0.45, 10))
plt.grid(True)

# # plot values
# t_plot_values = [0, 1, 3, 10, 20]
# t_plot_indices = [int(np.argmin(np.abs(ts - t))) for t in t_plot_values]
# # ^ indices of closest time values to the requested
# xs = xxs[0, :]
# for t_plot_index in t_plot_indices:
#     plt.figure()
#     plt.title(r'$t = {:.1f}$'.format(ts[t_plot_index]))
#     plt.plot(xs, A[t_plot_index, :], label=r'$a$')
#     plt.plot(xs, B[t_plot_index, :], label=r'$b$')
#     plt.plot(xs, C[t_plot_index, :], label=r'$c$')
#     plt.plot(xs, 1 - A[t_plot_index, :] - B[t_plot_index, :] - C[t_plot_index, :], label=r'$d$')
#     plt.xlabel(r'$x$')
#     plt.ylabel('Concentration')
#     plt.legend()

plt.savefig('figures/ece_with_comparison.pgf')
plt.savefig('figures/ece_with_comparison.png')

plt.show()
