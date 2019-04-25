import math
import numpy as np
from scipy.sparse import diags
from glob import glob
import os

path = 'data/ece/'


def solve_ece(alpha, ka_0, kc_0, k, E, E0a, E0c, delta_t, delta_x, t_max, x_max, save, label, force_calculation=False):
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
            xxs, tts, A, B, C = None, None, None, None, None
            for file_path in paths:
                pos = -5  # the position of the character 't', 'x' or 'U' telling us which matrix this is
                if file_path[pos] == 't':
                    tts = np.load(file_path)
                elif file_path[pos] == 'x':
                    xxs = np.load(file_path)
                elif file_path[pos] == 'A':
                    A = np.load(file_path)
                elif file_path[pos] == 'B':
                    B = np.load(file_path)
                elif file_path[pos] == 'C':
                    C = np.load(file_path)

            # make sure they all exist
            if tts is not None and xxs is not None and A is not None and B is not None and C is not None:
                return tts, xxs, A, B, C

    # solution meshes
    r = delta_t / (delta_x ** 2)
    N = math.ceil(t_max/delta_t)
    J = math.ceil(x_max/delta_x)
    ts = np.arange(0, t_max, delta_t)
    xs = np.arange(0, x_max, delta_x)
    xxs, tts = np.meshgrid(xs, ts)
    A = np.zeros_like(tts)
    A[0, :] = 1
    B = np.zeros_like(tts)
    C = np.zeros_like(tts)

    # finite difference matrix (changes for different t)
    def f_d_matrix(t_n):
        # exponents
        exp1a = np.exp((1-alpha)*(E(t_n)-E0a))
        exp1c = np.exp((1-alpha)*(E(t_n)-E0c))
        exp2a = np.exp(-alpha*(E(t_n)-E0a))
        exp2c = np.exp(-alpha*(E(t_n)-E0c))

        # form diagonals of finite difference matrix
        main_A_diagonal = (1 + 2*delta_t/(delta_x**2))*np.ones(J)
        main_A_diagonal[0] = 1 + 2*delta_t/(delta_x**2) + 2*(delta_t/delta_x)*ka_0*exp1a
        main_A_diagonal[-1] = 1

        main_B_diagonal = (1 + k*delta_t + 2*delta_t/(delta_x**2))*np.ones(J)
        main_B_diagonal[0] = 1 + k*delta_t + 2*delta_t/(delta_x**2) + 2*(delta_t/delta_x)*ka_0*exp2a
        main_B_diagonal[-1] = 1

        main_C_diagonal = (1 + 2*delta_t/(delta_x**2))*np.ones(J)
        main_C_diagonal[0] = 1 + 2*delta_t/(delta_x**2) + 2*(delta_t/delta_x)*kc_0*(exp1c + exp2c)
        main_C_diagonal[-1] = 1

        off_diagonal = -delta_t/(delta_x**2)*np.ones(J-1)  # this fits all, but we have to be careful

        C_B_diagonal = -k*delta_t*np.ones(J)
        C_B_diagonal[-1] = 0
        C_B_diagonal = np.concatenate((np.zeros(J), C_B_diagonal))

        # create matrix and edit the single values
        matrix = diags([
            np.concatenate((main_A_diagonal, main_B_diagonal, main_C_diagonal)),
            np.concatenate((off_diagonal, [0], off_diagonal, [0], off_diagonal)),
            np.concatenate((off_diagonal, [0], off_diagonal, [0], off_diagonal)),
            C_B_diagonal
        ], [0, -1, 1, -J]).toarray()
        matrix[0, 1] = -2*delta_t/(delta_x**2)
        matrix[J, J+1] = -2*delta_t/(delta_x**2)
        matrix[2*J, 2*J+1] = -2*delta_t/(delta_x**2)
        matrix[J-1, J-2] = 0
        matrix[2*J-1, 2*J-2] = 0
        matrix[3*J-1, 3*J-2] = 0
        matrix[J, 0] = -(2*(delta_t/delta_x)*ka_0*exp1a)
        matrix[2*J, 0] = 2*(delta_t/delta_x)*kc_0*exp2c
        matrix[0, J] = -(2*(delta_t/delta_x)*ka_0*exp2a)
        matrix[2*J, J] = 2*(delta_t/delta_x)*kc_0*exp2c - k*delta_t

        return matrix

    # solve
    for n in range(N - 1):
        # solve concentration
        last = np.concatenate((A[n, :], B[n, :], C[n, :])).copy()
        matrix = f_d_matrix(ts[n+1])
        next = np.linalg.solve(matrix, last +
                               2*(delta_t/delta_x)*kc_0*np.exp(-alpha*(E(ts[n+1])-E0c))*np.eye(1, 3*J, k=2*J)\
                               .reshape(-1))

        # load into solution arrays
        A[n+1, :] = next[0:J].copy()
        B[n+1, :] = next[J:2*J].copy()
        C[n+1, :] = next[2*J:3*J].copy()

    # save to file
    if save:
        # create directories if they don't exist
        if not os.path.exists('{}{}/dt={}_dx={}'.format(path, label, delta_t, delta_x)):
            os.makedirs('{}{}/dt={}_dx={}'.format(path, label, delta_t, delta_x))
        np.save('{}{}/dt={}_dx={}/t'.format(path, label, delta_t, delta_x), tts)
        np.save('{}{}/dt={}_dx={}/x'.format(path, label, delta_t, delta_x), xxs)
        np.save('{}{}/dt={}_dx={}/A'.format(path, label, delta_t, delta_x), A)
        np.save('{}{}/dt={}_dx={}/B'.format(path, label, delta_t, delta_x), B)
        np.save('{}{}/dt={}_dx={}/C'.format(path, label, delta_t, delta_x), C)

    # return
    return tts, xxs, A, B, C


def current_ece(tts, xxs, A, C):
    ts = tts[:, 0]
    xs = xxs[0, :]

    # we'll use a second-order (three point) scheme to find the current
    h_0 = xs[1] - xs[0]
    h_1 = xs[2] - xs[1]

    a = -(2*h_0 + h_1)/(h_0**2 + h_0*h_1)
    b = 1/h_0 + 1/h_1
    c = -h_0/(h_1**2 + h_0*h_1)

    I = a*(A[:, 0] + C[:, 0]) + b*(A[:, 1] + C[:, 1]) + c*(A[:, 2] + C[:, 2])

    return ts, I
