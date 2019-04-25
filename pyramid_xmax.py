import numpy as np
import matplotlib.pyplot as plt
from solve import solve, current

force_calculation = True

# model parameters
alpha = 0.5
k_0 = 35
E0 = 0
E_start = -10
E_end = 10
t_f = 20

# boundary conditions
u_0 = lambda x: np.full_like(x, 1)  # u(x, t=0)
u_Inf = 1  # u(x -> Inf, t>=0), we will be prescribing this at x_max instead

# numerical parameters
delta_t = 0.1
delta_x = 0.1
t_max = 2*t_f
x_maxs = np.arange(6, 13, 2)

# model functions
E = lambda t: E_end - np.abs(E_end - E_start - t*t_f/(E_end-E_start))

# solve
for x_max in x_maxs:
    tts, xxs, U = solve(alpha, k_0, E, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max,
                        True, 'pyramid/xmax', force_calculation, verbose=False)
    ts, I = current(tts, xxs, U)

    plt.plot(E(ts), I, label=r'$x_{\mathrm{max}}=' + str(x_max) + r'$')

plt.xlabel(r'Applied voltage $E$')
plt.xticks(np.linspace(E(0), E(t_f), 11))
plt.ylabel(r'Current $I$')
plt.grid(True)
plt.legend()
plt.savefig('figures/pyramid_xmax.png')
plt.savefig('figures/pyramid_xmax.pgf')
plt.show()
