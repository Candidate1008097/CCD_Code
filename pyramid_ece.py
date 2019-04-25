import numpy as np
import matplotlib.pyplot as plt
from solve_ece import solve_ece, current_ece

force_calculation = True

# model parameters
alpha = 0.5
ka_0 = 35
kc_0 = 35
k = 1
E0a = 0
E0c = 6
E_start = -10
E_end = 10
t_f = 20

# numerical parameters
delta_t = 0.1
delta_x = 0.1
t_max = 2*t_f
x_max = 6

# model functions
E = lambda t: E_end - np.abs(E_end - E_start - t*t_f/(E_end-E_start))

# solve
tts, xxs, A, B, C = solve_ece(alpha, ka_0, kc_0, k, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                              True, 'pyramid_ece', force_calculation)
ts, I = current_ece(tts, xxs, A, C)

# plot
# plt.plot(ts, I, 'k')
# plt.xlabel(r'Time $t$')
# plt.xticks(np.linspace(0, t_max, 11))
# plt.ylabel(r'Current $I$')
# plt.grid(True)
# plt.show()

# plot with reverse journey
plt.plot(E(ts), I, 'k')
plt.xlabel(r'Applied voltage $E$')
plt.xticks(np.linspace(E(0), E(t_f), 11))
plt.ylabel(r'Current $I$')
plt.grid(True)
plt.show()
