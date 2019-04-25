import math
import numpy as np
from glob import glob
import os

path = 'data/'


def solve(alpha, k_0, E, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max, save, label, force_calculation=False):
    """Solve the diffusion PDE describing the electrochemical reaction.

    :param alpha: A model parameter.
    :param k_0: A model parameter.
    :param E: A function of t describing the voltage sweep over time.
    :param E0: The base voltage value.
    :param u_0: A vectorised function of x describing the initial state of the system.
    :param u_Inf: A scalar value describing the concentration at x = Infinity.
    :param delta_t: The resolution of the time grid.
    :param delta_x: The resolution of the space grid.
    :param t_max: The maximum t-value.
    :param x_max: The maximum x-value (treated as infinity).
    :param save: Save to file? (Boolean)
    :param label: A string that will be used as the file name.
    :param force_calculation: If True, this function will always do the full calculation, even if it finds an existing
    file containing the data.
    :return tts, xxs, U: Three arrays of the same size, holding the t-value, x-value, and concentration respectively.
    """
    # check existence first
    if not force_calculation:
        paths = glob('{}{}/dt={}_dx={}/*'.format(path, label, delta_t, delta_x))
        if paths:
            xxs, tts, U = None, None, None
            for file_path in paths:
                pos = -5  # the position of the character 't', 'x' or 'U' telling us which matrix this is
                if file_path[pos] == 't':
                    tts = np.load(file_path)
                elif file_path[pos] == 'x':
                    xxs = np.load(file_path)
                elif file_path[pos] == 'U':
                    U = np.load(file_path)

            # make sure they all exist
            if tts is not None and xxs is not None and U is not None:
                return tts, xxs, U

    # solution meshes
    r = delta_t / (delta_x ** 2)
    J = math.ceil(t_max/delta_t)
    N = math.ceil(x_max/delta_x)
    ts = np.arange(0, t_max, delta_t)
    xs = np.arange(0, x_max, delta_x)
    xxs, tts = np.meshgrid(xs, ts)
    U = np.zeros_like(tts)
    U[0, :] = u_0(xs)

    # finite difference matrix (changes for different t)
    def f_d_matrix(t_n):
        matrix = - r * np.eye(N, k=-1) - r * np.eye(N, k=1) + (1 + 2 * r) * np.eye(N)
        matrix[0, 0] = 1 + 2 * r * (
                    1 + delta_x * k_0 * (math.exp(-alpha * (E(t_n) - E0)) + math.exp((1 - alpha) * (E(t_n) - E0))))
        matrix[0, 1] = -2 * r
        matrix[-1, -1] = 1
        matrix[-1, -2] = 0
        return matrix

    # solve
    for j in range(J - 1):
        # solve concentration
        U[j + 1, :] = np.linalg.solve(f_d_matrix(ts[j + 1]), U[j, :] +
                                      2 * r * delta_x * k_0 * math.exp((1 - alpha) *
                                                                       (E(ts[j + 1]) - E0)) * np.eye(1, N).reshape(-1))

    # save to file
    if save:
        # create directories if they don't exist
        if not os.path.exists('{}{}/dt={}_dx={}'.format(path, label, delta_t, delta_x)):
            os.makedirs('{}{}/dt={}_dx={}'.format(path, label, delta_t, delta_x))
        np.save('{}{}/dt={}_dx={}/t'.format(path, label, delta_t, delta_x), tts)
        np.save('{}{}/dt={}_dx={}/x'.format(path, label, delta_t, delta_x), xxs)
        np.save('{}{}/dt={}_dx={}/U'.format(path, label, delta_t, delta_x), U)

    # return
    return tts, xxs, U


def current(tts, xxs, U):
    """Find the current over time for the given concentration values.

    :param tts: The array of time values for each data point.
    :param xxs: As above, but for space.
    :param U: The concentration values at each point in space and time.
    :return ts, I: Two 1D arrays detailing the time and current at the electrode respectively.
    """
    # ts = tts[:, 0]
    # # take advantage of vectorised functions
    # E_t = E(ts) - E0
    # U_0 = U[:, 0]
    # exp0 = np.exp((1-alpha)*E_t)
    # exp1 = np.exp(-alpha*E_t)
    # I = k_0 * (U_0*exp0 - (1-U_0)*exp1)

    ts = tts[:, 0]
    xs = xxs[0, :]

    # we'll use a second-order (three point) scheme to find the current
    h_0 = xs[1] - xs[0]  # grid spacing, use to approximate current
    h_1 = xs[2] - xs[1]  # usually the same as h_0, but included so we can use it later for different spacings

    a = -(2*h_0 + h_1)/(h_0**2 + h_0*h_1)
    b = 1/h_0 + 1/h_1
    c = -h_0/(h_1**2 + h_0*h_1)

    I = a*U[:, 0] + b*U[:, 1] + c*U[:, 2]

    return ts, I
