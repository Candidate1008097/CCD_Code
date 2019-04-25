from solve_ece import solve_ece, current_ece
from solve_ec2e import solve_ec2e, current_ec2e
import numpy as np
import matplotlib.pyplot as plt

force_calculation = True  # has to be true or it won't update

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

# model functions
E = lambda t: t + E_start

# offset to help with the glitch at the beginning
offset = 2
0.975
# set up iteration
E0a = 0
E0cs = np.linspace(0, 10, 50)
peaks_ece = []
peaks_ec2e = []
for i, E0c in enumerate(E0cs):
    # print E0c
    print("E0c =", E0c)
    print("{} out of {}".format(i+1, len(E0cs)))
    print()

    # solve
    tts1, xxs1, A1, B1, C1 = solve_ece(alpha, ka_0, kc_0, k, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                                       True, 'peak_split/ece', force_calculation)
    ts1, I1 = current_ece(tts1, xxs1, A1, C1)
    tts2, xxs2, A2, B2, C2 = solve_ec2e(alpha, ka_0, kc_0, k, E, E0a, E0c, delta_t, delta_x, t_max, x_max,
                                        True, 'peak_split/ec2e', force_calculation)
    ts2, I2 = current_ec2e(tts2, xxs2, A2, C2)

    # find peak(s) by iterating through to find local maxima
    peak1_ece = None
    peak2_ece = None
    peak1_ec2e = None
    peak2_ec2e = None
    for j in range(len(I1) - offset - 1):
        # is this a local maximum?
        if I1[j+offset] > I1[j+offset-1] and I1[j+offset] > I1[j+offset+1]:
            # it's a peak
            if peak1_ece is None:
                peak1_ece = I1[j+offset]
            elif peak2_ece is None:
                peak2_ece = I1[j+offset]
            else:
                # something odd has happened
                print("Third peak found?")
                print("Time t1 =", ts1[j+offset])
                print("Current values = {}, {}, {}".format(I1[j+offset-1], I1[j+offset], I1[j+offset+1]))

        # again for ec2e
        if I2[j+offset] > I2[j+offset-1] and I2[j+offset] > I2[j+offset+1]:
            # it's a peak
            if peak1_ec2e is None:
                peak1_ec2e = I2[j+offset]
            elif peak2_ec2e is None:
                peak2_ec2e = I2[j+offset]
            else:
                # something odd has happened
                print("Third peak found?")
                print("Time t2 =", ts2[j+offset])
                print("Current values = {}, {}, {}".format(I2[j+offset-1], I2[j+offset], I2[j+offset+1]))
    peaks_ece.append((peak1_ece, peak2_ece))
    peaks_ec2e.append((peak1_ec2e, peak2_ec2e))

# split into upper and lower peaks
upper_ece, lower_ece = [list(t) for t in zip(*peaks_ece)]
upper_ec2e, lower_ec2e = [list(t) for t in zip(*peaks_ec2e)]

# transfer the latter parts from the upper to the lower ones
reached = False
for i, t in reversed(list(enumerate(zip(upper_ece, lower_ece)))):
    if t[1] is not None:
        reached = True
    if t[1] is None and reached:
        reached = False
        break
    upper_ece[i], lower_ece[i] = lower_ece[i], upper_ece[i]
for i, t in reversed(list(enumerate(zip(upper_ec2e, lower_ec2e)))):
    if t[1] is not None:
        reached = True
    if t[1] is None and reached:
            break
    upper_ec2e[i], lower_ec2e[i] = lower_ec2e[i], upper_ec2e[i]

# plot
plt.plot(E0cs, upper_ece, 'C0', label=r'$ECE$')
plt.plot(E0cs, lower_ece, 'C0')
plt.plot(E0cs, upper_ec2e, 'C1', label=r'$EC_2E$')
plt.plot(E0cs, lower_ec2e, 'C1')
plt.xlabel(r'$E_0^c$')
plt.ylabel(r'Time $t$')
plt.grid(True)
plt.legend()
plt.savefig('figures/peak_split.png')
plt.savefig('figures/peak_split.pgf')
plt.show()
