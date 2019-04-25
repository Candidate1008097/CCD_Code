from solve_ece import solve_ece, current_ece
import matplotlib
import matplotlib.pyplot as plt

save = True
force_calculation = False

# model parameters
alpha = 0.5
ka_0 = 35
kc_0 = 35
k = 1
E_start = -10

# numerical parameters
delta_t = 0.1
delta_x = 0.1
t_max = 20
x_max = 6

# model function
E = lambda t: t + E_start

# different values of E0 to test
E0as = [-4, -2, 0, 2, 4]
E0cs = [-4, -2, 0, 2, 4]

plt.figure(figsize=(4.768, 4.768))
plt.subplots_adjust(top=0.9, bottom=0.025, left=0.025, right=0.975, wspace=0.1, hspace=0.7)

for i, E0a in enumerate(E0as):
    for j, E0c in enumerate(E0cs):
        print('{} of {}...'.format(len(E0cs)*i+j+1, len(E0as)*len(E0cs)), end='')

        # solve
        tts, xxs, a, _, c = solve_ece(alpha, ka_0, kc_0, k, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                                      save, 'e0_variation/a={},c={}'.format(E0a, E0c), force_calculation)
        ts, current = current_ece(tts, xxs, a, c)

        # plot
        plt.subplot(len(E0cs), len(E0as), len(E0cs)*i+j+1)
        plt.gca().set_title(r'$E^0_a={}$, $E^0_c={}$'.format(E0a, E0c), fontsize=7)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.plot(ts, current, 'k', linewidth=0.6)

        print('done.')

plt.savefig('figures/ece_e0_variation.pgf')
plt.savefig('figures/ece_e0_variation.png')

plt.show()
