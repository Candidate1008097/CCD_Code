import numpy as np
import matplotlib.pyplot as plt
from solve import solve, current
from solve_ece import solve_ece, current_ece
from solve_ec2e import solve_ec2e, current_ec2e

force_calculation = True

# model parameters
alpha = 0.5
ka_0 = 35
kc_0 = 35
k = 1
E0a = -2
E0c = 4
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

# solve all
# tts1, xxs1, A1 = solve(alpha, ka_0, E, E0a, lambda x: 1, 1, delta_t, delta_x, t_max, x_max,
#                        True, 'pyramid_all', force_calculation)
tts2, xxs2, A2, B2, C2 = solve_ece(alpha, ka_0, kc_0, k, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                                   True, 'pyramid_all', force_calculation)
tts3, xxs3, A3, B3, C3 = solve_ec2e(alpha, ka_0, kc_0, k, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                                    True, 'pyramid_all', force_calculation)
# ts1, I1 = current(tts1, xxs1, A1)
ts2, I2 = current_ece(tts2, xxs2, A2, C2)
ts3, I3 = current_ec2e(tts3, xxs3, A3, C3)

# plot with reverse journey
# plt.plot(E(ts1), I1, label='E reaction')
plt.plot(E(ts2), I2, label=r'$ECE$ reaction')
plt.plot(E(ts3), I3, label=r'$EC_2E$ reaction')
plt.xlabel(r'Applied voltage $E$')
plt.xticks(np.linspace(E(0), E(t_f), 11))
plt.ylabel(r'Current $I$')
plt.grid(True)
plt.legend()
plt.savefig('figures/pyramid_2peak.png')
plt.savefig('figures/pyramid_2peak.pgf')
plt.show()
