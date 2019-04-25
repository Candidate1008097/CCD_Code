# CCD_Code
The code accompanying the dissertation "Numerical Solution of Problems in Electrochemistry".

Most of these files create figures used in the dissertation, some of them create unused figures. The three most important files are `solve.py`, `solve_ece.py`, and `solve_ec2e.py`. Each of these contains a function for performing the numerical solution, and a function for extracting the measured current at the electrode from this solution, as detailed in the dissertation itself.

For the most basic usage, see the file `implicit_euler_solve.py`.

Example usage:
```python
import numpy as np
from solve import solve, current

force_calculation = True

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
x_max = 6

# model functions
E = lambda t: t + E_start

# solve
tts, xxs, U = solve(alpha, k_0, E, E0, u_0, u_Inf, delta_t, delta_x, t_max, x_max, True, 'check', force_calculation)
ts, I = current(tts, xxs, U)
```
then plot using your preferred plotting library.
