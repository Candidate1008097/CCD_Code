import numpy as np
import matplotlib.pyplot as plt
from solve import solve, current

force_calculation = False

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
delta_ts = [0.1, 0.01, 0.001, 0.1, 0.01, 0.001]
delta_xs = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
t_max = 2*t_f
x_max = 6

# model functions
E = lambda t: E_end - np.abs(E_end - E_start - t*t_f/(E_end-E_start))

# solve
for delta_t, delta_x in zip(delta_ts, delta_xs):
    print('Delta t = {}, Delta x = {}'.format(delta_t, delta_x))
    tts, xxs, U = solve(alpha, k_0, E, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max,
                        True, 'pyramid', force_calculation, verbose=True)
    ts, I = current(tts, xxs, U)

    plt.plot(E(ts), I, label=r'$h={}$, $k={}$'.format(delta_x, delta_t))

plt.xlabel(r'Applied voltage $E$')
plt.xticks(np.linspace(E(0), E(t_f), 11))
plt.ylabel(r'Current $I$')
plt.grid(True)
plt.legend()
plt.savefig('figures/pyramid_error.png')
plt.savefig('figures/pyramid_error.pgf')
plt.show()
