import math
import numpy as np
import matplotlib.pyplot as plt
from solve import solve, current

# calculate from scratch?
force_calculation = False

# model parameters
alpha = 0.5
k_0 = 35
E0 = 0
E_start = -10
theta = 38.92

# boundary conditions
u_0 = lambda x: 1  # u(x, t=0)
u_Inf = 1  # u(x -> Inf, t>=0), we will be prescribing this at x_max instead

# variable conditions
omega_1 = math.pi
omega_2 = 4*math.pi
delta_E_1 = 0.4#/theta
delta_E_2 = 0.1#/theta

# three different functions
def E_dc(t):
    return E_start + t

def E_ac_1(t):
    return E_start + t + delta_E_1*math.sin(omega_1*t)

def E_ac_2(t):
    return E_start + t + delta_E_2*math.sin(omega_2*t)

# numerical parameters
delta_t = 0.001  # so there are >100 timesteps per oscillation
delta_x = 0.1
t_max = 20
x_max = 6

# solve
tts, xxs, U_dc = solve(alpha, k_0, E_dc, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max,
                       True, 'single_ac_base', force_calculation)
ts, I_dc = current(tts, xxs, U_dc)
_, _, U_ac_1 = solve(alpha, k_0, E_ac_1, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max,
                     True, 'single_ac_1', force_calculation)
_, I_ac_1 = current(tts, xxs, U_ac_1)
_, _, U_ac_2 = solve(alpha, k_0, E_ac_2, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max,
                     True, 'single_ac_2', force_calculation)
_, I_ac_2 = current(tts, xxs, U_ac_2)

# plot 4 subplots
plt.figure(figsize=(4.768, 4))
plt.subplot(221)
plt.plot(ts, I_ac_1, 'k', linewidth=0.6)
plt.xlabel(r'Time $t$', fontsize='small')
plt.ylabel(r'Current $I$', fontsize='small')
plt.yticks(np.arange(0, 0.7, 0.1))
plt.title(r'$\Omega = \pi, \Delta E = 0.4$')

plt.subplot(222)
plt.plot(ts, I_ac_1 - I_dc, 'k', linewidth=0.6)
plt.xlabel(r'Time $t$', fontsize='small')
plt.ylabel(r'AC current $I$', fontsize='small')
plt.yticks(np.linspace(-0.2, 0.2, 5))
plt.title(r'$\Omega = \pi, \Delta E = 0.4$')

plt.subplot(223)
plt.plot(ts, I_ac_2, 'k', linewidth=0.6)
plt.xlabel(r'Time $t$', fontsize='small')
plt.ylabel(r'Current $I$', fontsize='small')
plt.yticks(np.arange(0, 0.7, 0.1))
plt.title(r'$\Omega = 4\pi, \Delta E = 0.1$')

plt.subplot(224)
plt.plot(ts, I_ac_2 - I_dc, 'k', linewidth=0.6)
plt.xlabel(r'Time $t$', fontsize='small')
plt.ylabel(r'AC current $I$', fontsize='small')
plt.yticks(np.arange(-0.1, 0.11, 0.05))
plt.title(r'$\Omega = 4\pi, \Delta E = 0.1$')

plt.subplots_adjust(hspace=1, wspace=0.6)

# save plots
plt.savefig('figures/implicit_euler_solve_ac.pgf', bbox_inches='tight')
plt.savefig('figures/implicit_euler_solve_ac.png', bbox_inches='tight')

plt.show()
