from solve_ec2e import solve_ec2e, current_ec2e
import numpy as np
import matplotlib.pyplot as plt

force_calculation = True

# model parameters
alpha = 0.5
ka_0 = 35
kc_0 = 35
k = 1
E0a = 0
E0c = 4
E_start = -10

# numerical parameters
delta_t = 0.1
delta_x = 0.1
t_max = 20
x_max = 6

# model functions
E = lambda t: t + E_start

# solve
tts, xxs, A, B, C = solve_ec2e(alpha, ka_0, kc_0, k, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                               True, 'ec2e', force_calculation)
ts, I = current_ec2e(tts, xxs, A, C)

# plot
plt.plot(ts, I, 'k', label='Numerical simulation')
plt.xlabel(r'$t$ (time)')
plt.xticks(np.linspace(0, 20, 11))
plt.ylabel(r'$I$ (current)')
# plt.yticks(np.linspace(0, 0.45, 10))
plt.grid(True)

# plot values
t_plot_values = [0, 1, 3, 10, 20]
t_plot_indices = [int(np.argmin(np.abs(ts - t))) for t in t_plot_values]
# ^ indices of closest time values to the requested
xs = xxs[0, :]
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

plt.show()
