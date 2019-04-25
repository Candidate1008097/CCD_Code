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
x_max1 = 6
x_max2 = 20

# model functions
E = lambda t: t + E_start

# solve
tts1, xxs1, U1 = solve(alpha, k_0, E, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max1,
                       True, 'comparison/6', force_calculation)
ts1, I1 = current(tts1, xxs1, U1)
tts2, xxs2, U2 = solve(alpha, k_0, E, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max2,
                       True, 'comparison/20', force_calculation)
ts2, I2 = current(tts2, xxs2, U2)

# plot
plt.figure(figsize=(4.768, 4.768))

plt.subplot(2, 1, 1)
plt.plot(ts1, I1, label='$x_{\mathrm{max}}=6$')
plt.plot(ts2, I2, label='$x_{\mathrm{max}}=20$')

plt.legend()
plt.xlabel(r'$t$ (time)')
plt.xticks(np.linspace(0, 20, 11))
plt.ylabel(r'$I$ (current)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(ts1, I1 - I2)
plt.xlabel(r'$t$ (time)')
plt.xticks(np.linspace(0, 20, 11))
plt.ylabel(r'Difference')
plt.grid(True)

# save plots
plt.savefig('figures/comparison.png')
plt.savefig('figures/comparison.pgf')

plt.show()
