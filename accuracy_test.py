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
delta_ts = [0.1, 0.01, 0.001, 0.1, 0.01, 0.001]
delta_xs = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
t_max = 20
x_max = 6

# model functions
E = lambda t: t + E_start

ends = []

for delta_t, delta_x in zip(delta_ts, delta_xs):
    # solve
    tts, xxs, U = solve(alpha, k_0, E, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max,
                        True, 'acc', force_calculation, verbose=True)
    ts, I = current(tts, xxs, U)
    ends.append(I[-1])

    # plot
    plt.plot(ts, I, label=r'$h={}$, $k={}$'.format(delta_x, delta_t))

ends = np.array(ends)
diff = ends.max() - ends.min()
print('Difference =', diff)

plt.legend()
plt.xlabel(r'$t$ (time)')
plt.xticks(np.linspace(0, 20, 11))
plt.ylabel(r'$I$ (current)')
plt.yticks(np.linspace(0, 0.45, 10))
plt.grid(True)

# save plots
plt.savefig('figures/accuracy.pgf')
plt.savefig('figures/accuracy.png')

plt.show()
