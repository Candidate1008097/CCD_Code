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
E0a = -2
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
tts1, xxs1, A1, B1, C1 = solve_ece(alpha, ka_0, kc_0, k, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                                   True, 'test', force_calculation)
tts2, xxs2, A2, B2, C2 = solve_ec2e(alpha, ka_0, kc_0, k, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                                    True, 'test', force_calculation)
ts1, I1 = current_ece(tts1, xxs1, A1, C1)
ts2, I2 = current_ec2e(tts2, xxs2, A2, C2)

# plot
plt.plot(ts1, I1, label=r'ECE')
plt.plot(ts2, I2, label=r'EC$_2$E')
plt.xlabel(r'$t$ (time)')
plt.xticks(np.linspace(0, 20, 11))
plt.ylabel(r'$I$ (current)')
# plt.yticks(np.linspace(0, 0.45, 10))
plt.legend()
plt.grid(True)
plt.show()
