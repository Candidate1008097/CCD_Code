from solve import solve, current
from solve_ece import solve_ece, current_ece
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
tts, xxs, A = solve(alpha, ka_0, E, E0a, lambda x: 1, 1, delta_t, delta_x, t_max, x_max,
                    True, 'all_3/orig', force_calculation)
tts_ece, xxs_ece, A_ece, B_ece, C_ece = solve_ece(alpha, ka_0, kc_0, k, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                                                  True, 'all_3/ece', force_calculation)
tts_ec2e, xxs_ec2e, A_ec2e, B_ec2e, C_ec2e = solve_ec2e(alpha, ka_0, kc_0, k, E, E0a, E0c, delta_t, delta_x, t_max,
                                                        x_max, True, 'all_3/ec2e', force_calculation)
ts, I = current(tts, xxs, A)
ts_ece, I_ece = current_ece(tts_ece, xxs_ece, A_ece, C_ece)
ts_ec2e, I_ec2e = current_ec2e(tts_ec2e, xxs_ec2e, A_ec2e, C_ec2e)

# plot
plt.plot(ts_ec2e, I_ec2e, label=r'$EC_2E$ reaction')
plt.plot(ts_ece, I_ece, '--', label=r'$ECE$ reaction')
plt.plot(ts, I, '--', label=r'Standard reaction')
plt.xlabel(r'$t$ (time)')
plt.xticks(np.linspace(0, 20, 11))
plt.ylabel(r'$I$ (current)')
plt.grid(True)
plt.legend()

plt.savefig('figures/all_3.png')
plt.savefig('figures/all_3.pgf')

plt.show()
