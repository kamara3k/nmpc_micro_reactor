

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define constants and initial parameters
nx = 12  # Number of states
ny = 1   # Number of outputs
nu = 1   # Number of control inputs
dt = 0.2  # Time step for simulation
Ts = 0.02  # Time step for control

# MPC parameters
PredictionHorizon = 10
ControlHorizon = 3

# Define number of drums
num_drums = 1

# Simulation parameters
T = 2000
time = np.arange(0, T + dt, dt)
nt = len(time)
ref = np.zeros(nt)

# Function to set parameters based on number of drums
def setParameters(num_drums):
    if num_drums == 8:
        Rho_d0 = -0.033085599
        Reactivity_per_degree = 26.11e-5
        u0 = 77.56
    elif num_drums == 4:
        Rho_d0 = -0.033085599
        Reactivity_per_degree = 16.11e-5
        u0 = 125.56
    elif num_drums == 2:
        Rho_d0 = -0.033085599 + 0.0073
        Reactivity_per_degree = 7.33e-5
        u0 = 177.84
    elif num_drums == 1:
        Rho_d0 = -0.033085599 + 0.0078 + 0.0082
        Reactivity_per_degree = 2.77e-5
        u0 = 177.84
    else:
        Rho_d0 = -0.033085599
        Reactivity_per_degree = 26.11e-5
        u0 = 77.56
    return Rho_d0, Reactivity_per_degree, u0

Rho_d0, Reactivity_per_degree, u0 = setParameters(num_drums)

# Generate reference trajectory
time_point = np.array([0, 20, 30, 50, 60, 80, 90, 110, 130, 200]) * 10
pow = np.array([1, 1, 0.4, 0.4, 1, 1, 0.4, 0.4, 1, 1])
ref_old = pow[0]
for it in range(nt):
    if it > 0:
        ref_old = ref[it - 1]
    ref[it] = ref_old
    for ii in range(len(time_point) - 1):
        if time_point[ii] <= time[it] <= time_point[ii + 1]:
            frac1 = (time_point[ii + 1] - time[it]) / (time_point[ii + 1] - time_point[ii])
            frac2 = 1.0 - frac1
            ref[it] = frac1 * pow[ii] + frac2 * pow[ii + 1]
            break

# Initial conditions
Sig_x = 2.65e-22
yi = 0.061
yx = 0.002
lamda_x = 2.09e-5
lamda_I = 2.87e-5 
Sum_f = 0.3358
G = 3.2e-11
V = 400 * 200
P_0 = 22e6
Pi = P_0 / (G * Sum_f * V)
Xe0 = (yi + yx) * Sum_f * Pi / (lamda_x + Sig_x * Pi)
I0 = yi * Sum_f * Pi / lamda_I

# Initialize state and manipulated variable
x0 = np.array([pow[0], pow[0], pow[0], pow[0], pow[0], pow[0], pow[0], I0, Xe0, 900.42, 898.28, 888.261])
x = x0
y = x0[0]
mv = u0

# Initialize history storage
xHistory = np.zeros((nx, len(time)))
MV = np.zeros(len(time))
xHistory[:, 0] = x0
MV[0] = u0

# CasADi symbolic variables
x_sym = ca.MX.sym('x', nx)
u_sym = ca.MX.sym('u', nu)

# Define reactor dynamics using CasADi, maintaining symbolic formulation (no hardcoded numbers)
def reactorCT0_casadi(x, u, Rho_d0, Reactivity_per_degree):
    dx = ca.MX.zeros(nx)
    
    # Extract state variables
    n_r, Cr1, Cr2, Cr3, Cr4, Cr5, Cr6, X, I, Tf, Tm, Tc = ca.vertsplit(x)

    # Define reactor dynamics variables (same as in your code)
    Sig_x = 2.65e-22
    yi = 0.061
    yx = 0.002
    lamda_x = 2.09e-5
    lamda_I = 2.87e-5
    Sum_f = 0.3358
    l = 1.68e-3
    beta = 0.0048
    beta_1 = 1.42481e-4
    beta_2 = 9.24281e-4
    beta_3 = 7.79956e-4
    beta_4 = 2.06583e-3
    beta_5 = 6.71175e-4
    beta_6 = 2.17806e-4
    Lamda_1 = 1.272e-2
    Lamda_2 = 3.174e-2
    Lamda_3 = 1.16e-1
    Lamda_4 = 3.11e-1
    Lamda_5 = 1.4
    Lamda_6 = 3.87

    cp_f = 977
    cp_m = 1697
    cp_c = 5188.6
    M_f = 2002
    M_m = 11573
    M_c = 500
    mu_f = M_f * cp_f
    mu_m = M_m * cp_m
    mu_c = M_c * cp_c
    f_f = 0.96
    P_0 = 22e6
    Tf0 = 1105
    Tm0 = 1087
    T_in = 864
    T_out = 1106
    Tc0 = (T_in + T_out) / 2
    K_fm = f_f * P_0 / (Tf0 - Tm0)
    K_mc = P_0 / (Tm0 - Tc0)
    M_dot = 1.75e1
    alpha_f = -2.875e-5
    alpha_m = -3.696e-5
    alpha_c = 0.0
    X0 = 2.35496411413791e10

    # Calculate reactivity and dynamics
    Rho_d1 = Rho_d0 + u * Reactivity_per_degree
    G = 3.2e-11
    V = 400 * 200
    Pi = P_0 / (G * Sum_f * V)
    rho = (Rho_d1 + alpha_f * (Tf - Tf0) + alpha_c * (Tc - Tc0) +
           alpha_m * (Tm - Tm0) - Sig_x * (X - X0) / Sum_f)

    # Power dynamics
    dx[0] = (rho - beta) / l * n_r + beta_1 / l * Cr1 + beta_2 / l * Cr2 + beta_3 / l * Cr3 + \
            beta_4 / l * Cr4 + beta_5 / l * Cr5 + beta_6 / l * Cr6
    dx[1] = Lamda_1 * n_r - Lamda_1 * Cr1
    dx[2] = Lamda_2 * n_r - Lamda_2 * Cr2
    dx[3] = Lamda_3 * n_r - Lamda_3 * Cr3
    dx[4] = Lamda_4 * n_r - Lamda_4 * Cr4
    dx[5] = Lamda_5 * n_r - Lamda_5 * Cr5
    dx[6] = Lamda_6 * n_r - Lamda_6 * Cr6

    # Xenon and iodine dynamics
    dx[7] = yx * Sum_f * Pi + lamda_I * I - Sig_x * X * Pi - lamda_x * X
    dx[8] = yi * Sum_f * Pi - lamda_I * I

    # Thermal dynamics
    dx[9] = f_f * P_0 / mu_f * n_r - K_fm / mu_f * (Tf - Tc)
    dx[10] = (1 - f_f) * P_0 / mu_m * n_r + (K_fm * (Tf - Tm) - K_mc * (Tm - Tc)) / mu_m
    dx[11] = K_mc * (Tm - Tc) / mu_c - 2 * M_dot * cp_c * (Tc - T_in) / mu_c

    return dx

# Discretization of the system using RK4 method
def reactorDT0_casadi(x, u, Ts):
    M = 5  # Number of integration steps
    delta = Ts / M
    xk1 = x
    for _ in range(M):
        f1 = reactorCT0_casadi(xk1, u, Rho_d0, Reactivity_per_degree)
        hxk1 = xk1 + delta * f1
        f2 = reactorCT0_casadi(hxk1, u, Rho_d0, Reactivity_per_degree)
        xk1 = xk1 + delta * (f1 + f2) / 2
    return xk1

# CasADi function for system dynamics
reactor_f = ca.Function('reactor_f', [x_sym, u_sym], [reactorDT0_casadi(x_sym, u_sym, Ts)])

# NMPC objective function
def nmpc_objective(u, xk, ref, PredictionHorizon, ControlHorizon):
    J = 0
    x = xk
    Q = 1e7 # penalty on the tracking or error of the output state
    R = 0.1 # penalty on control input (drum position)
    Rd = 1 # penalty on rate of change of control drum

    for k in range(PredictionHorizon):
        u_k = u[k if k < ControlHorizon else ControlHorizon-1]
        x = reactor_f(x, u_k)
        y = x[0]  # Output (core power)
        J += Q * (y - ref)**2
        if k < ControlHorizon:
            J += R * u[k]**2
            if k > 0:
                J += Rd * (u[k] - u[k-1])**2
    return J

# NMPC optimization using CasADi
def manualNMPC_casadi(xk, mv, ref, PredictionHorizon, ControlHorizon, Ts, u_lb, u_ub, du_lb, du_ub):
    u_seq = ca.MX.sym('u_seq', ControlHorizon)  # Control sequence to optimize
    obj = nmpc_objective(u_seq, xk, ref, PredictionHorizon, ControlHorizon)  # Objective function

    # Set up CasADi solver
    solver = ca.nlpsol('solver', 'ipopt', {'x': u_seq, 'f': obj})   
    
    # In CasADi, solver is initialized as a variable, but it's an object that acts like a callable function.
    
    # Initial guess for control sequence
    u0_seq = np.ones(ControlHorizon) * mv

    # Solve the optimization problem
    sol = solver(x0=u0_seq, lbx=u_lb, ubx=u_ub)
    u_opt = sol['x'].full().flatten()

    return u_opt[0]

# Main simulation loop
for ct in range(1, len(time)):
    # Solve NMPC optimization
    mv = manualNMPC_casadi(x, mv, ref[ct], PredictionHorizon, ControlHorizon, Ts, 0, 180, -1 * dt, 1 * dt)

    # Update state using CasADi system dynamics
    x = reactor_f(x, mv).full().flatten()

    # Store history
    xHistory[:, ct] = x
    MV[ct] = mv

# Post-processing: compute and display MAE
error = xHistory[0, :] - ref
MAE = np.mean(np.abs(error))
print(f'Mean Absolute Error (MAE): {MAE:.8f}')

# Plotting results
t = np.arange(0, T + dt, dt)
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, xHistory[0, :] * 100, label='Actual power', linewidth=2)
plt.plot(t, ref * 100, '--', label='Desired power', linewidth=2)
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Power (%)')
plt.title('NMPC Microreactor System Core Power Simulation with 1 control drum')
plt.ylim([0, 200])
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, MV, linewidth=2)
plt.ylabel('Manipulated Variable')
plt.xlabel('Time (s)')
plt.ylim([110, 185])

plt.subplot(3, 1, 3)
plt.plot(t[1:], np.diff(MV) / dt, linewidth=2)
plt.ylim([-1.5, 1.5])
plt.ylabel('Rate of Change of MV')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()
