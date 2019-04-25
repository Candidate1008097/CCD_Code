from solve_ec2e import solve_ec2e, current_ec2e
import numpy as np
import matplotlib.pyplot as plt

force_calculation = False

# model parameters
alpha = 0.5
ka_0 = 35
kc_0 = 35
k = 1
E0a = 0
E0c = 4
E_start = -10

# numerical parameters
delta_ts = [0.1, 0.01, 0.001, 0.1, 0.01]
delta_xs = [0.1, 0.1, 0.1, 0.01, 0.01]
t_max = 20
x_max = 6

# model functions
E = lambda t: t + E_start

ends = []

for delta_t, delta_x in zip(delta_ts, delta_xs):
    # solve
    tts, xxs, A, B, C = solve_ec2e(alpha, ka_0, kc_0, k, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                                   True, 'ec2e/acc', force_calculation, verbose=True)
    ts, I = current_ec2e(tts, xxs, A, C)
    ends.append(I[-1])

    # plot
    plt.plot(ts, I, label=r'$h={}$, $k={}$'.format(delta_x, delta_t))

ends = np.array(ends)
diff = ends.max() - ends.min()
print('Difference at end =', diff)

plt.xlabel(r'$t$ (time)')
plt.xticks(np.linspace(0, 20, 11))
plt.ylabel(r'$I$ (current)')
plt.grid(True)
plt.legend()

plt.savefig('figures/accuracy_ec2e.png')
plt.savefig('figures/accuracy_ec2e.pgf')

plt.show()
