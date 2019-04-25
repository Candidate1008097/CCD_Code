import numpy as np
import matplotlib.pyplot as plt
from solve_ece import solve_ece, current_ece
from solve_ec2e import solve_ec2e, current_ec2e

force_calculation = False

# model parameters
alpha = 0.5
ka_0 = 35
kc_0 = 35
r = 1
E0a = 0
E0cs = np.arange(-4, 11, 6)
E_start = -10

# numerical parameters
delta_t = 0.1
delta_x = 0.1
t_max = 20
x_max = 6

# model functions
E = lambda t: t + E_start

# solve for all alphas
for i, E0c in enumerate(E0cs):
    tts_ece, xxs_ece, A_ece, B_ece, C_ece = solve_ece(alpha, ka_0, kc_0, r, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                                                      True, 'vary/E0c/E0c={}'.format(E0c), force_calculation)
    tts_ec2e, xxs_ec2e, A_ec2e, B_ec2e, C_ec2e = solve_ec2e(alpha, ka_0, kc_0, r, E, E0a, E0c, delta_t, delta_x, t_max,
                                                            x_max, True, 'vary/E0c_ec2e/E0c={}'.format(E0c),
                                                            force_calculation)
    ts_ece, I_ece = current_ece(tts_ece, xxs_ece, A_ece, C_ece)
    ts_ec2e, I_ec2e = current_ec2e(tts_ec2e, xxs_ec2e, A_ec2e, C_ec2e)

    # plot
    plt.plot(ts_ec2e, I_ec2e, label=r'$E_0^c={}$'.format(E0c))
    plt.plot(ts_ece, I_ece, 'C{}--'.format(i))

plt.legend()
plt.xlabel(r'$t$ (time)')
plt.xticks(np.linspace(0, 20, 11))
plt.ylabel(r'$I$ (current)')
plt.grid(True)

# save plots
plt.savefig('figures/vary_E0c_ec2e.pgf')
plt.savefig('figures/vary_E0c_ec2e.png')

plt.show()
