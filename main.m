
%% Parameters
% Constants
global g m_b m_p k_c dt;
g = 9.81;   % [m/s^2]
k_c = 10;   % [1/s]  time-constant of velocity controller

% Params
m_b = 0.03; % [kg]
m_p = 100;    % [kg]

% Initial Conditions
x_b0 = 0;   % [m]   starting ball position
x_p0 = 0;   % [m]   starting plate position
u_p0 = 0;  % [m/s] starting plate velocity
u_b0 = u_p0;   % [m/s] starting ball velocity


% Design params
dt = 0.01;                    % [s] discretization timestep
h_b_max = 1;                  % [m] maximal height the ball achievs
T = 2 * sqrt(8*h_b_max/g);    % [s] time for one iteration T = 2 T_b
N = 5*ceil(T / dt);           % number of steps for one iteration (mayve use floor)

%% Simulation Example
% Input
A = 0.3;                                    % [m] amplitude
timesteps = dt * (0:N);                     % [s,s,..] timesteps
F_p = 100 * m_p * A*sin(2*pi/T *timesteps); % [N] input force on the plate

% Simulation for N steps
[x_b, u_b, x_p, u_p, dP_N_vec, gN_vec] = Simulation.simulate_one_iteration(dt, N, x_b0, x_p0, u_b0, u_p0, F_p);

% Plotting for Simulation Example
close all
Simulation.plot_results(dt, F_p, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)

%% Desired Trajectory planning example
% Initialize disturbances
    d1 = 0;
    d2 = 0;

% Initialize throw and catch point
    h_b_max = 1; % [m] maximal height the ball achievs
    x_p0 = 0;
    x_pTb = x_p0;
    ap_0 = 0;
    ap_T = 0;

% A] Ball Height and Time
    [Tb, ub_0] = plan_ball_trajectory(h_b_max, d1, d2);

% B] Plate Trajectory
    close all
    % 1) Free start and end acceleration
    [xp_des, T] = MinJerkTrajectory.plan_plate_trajectory(Tb, x_p0, x_pTb, ub_0, -ub_0); 
    MinJerkTrajectory.plot_paths(xp_des, T, dt, 'Free start and end acceleration')

    % 2) Free end acceleration 
    [xp_des, T] = MinJerkTrajectory.plan_plate_trajectory(Tb, x_p0, x_pTb, ub_0, -ub_0, ap_0); 
    MinJerkTrajectory.plot_paths(xp_des, T, dt, 'Free end acceleration')

    % 3) Set start and end acceleration 
    [xp_des, T] = MinJerkTrajectory.plan_plate_trajectory(Tb, x_p0, x_pTb, ub_0, -ub_0, ap_0, ap_T); 
    MinJerkTrajectory.plot_paths(xp_des, T, dt, 'Set start and end acceleration')

%% Feedforward controled system
% Design params
h_b_max = 1;                  % [m] maximal height the ball achievs
T = 2 * sqrt(8*h_b_max/g);    % [s] time for one iteration T = 2 T_b
N = 5*ceil(T / dt);           % number of steps for one iteration (mayve use floor)

% Initialize disturbances
d1 = 0;
d2 = 0;
dup = 0;

% Initialize throw point
x_b0 = 0;
x_p0 = x_b0;
x_pTb = x_p0;

for j = j:N
    % 0. Compute Ball Height and Time
    [Tb, ub_0] = plan_ball_trajectory(h_b_max, d1, d2);
    
    % 1. Plan desired Plate trajectory (min jerk trajectory)
    [xp_des, T] = MinJerkTrajectory.plan_plate_trajectory(Tb, x_p0, x_pTb, ub_0, -ub_0);
%     [xp_des, T] = MinJerkTrajectory.plan_plate_trajectory(Tb, x_p0, x_pTb, ub_0, -ub_0, ap_0);
%     [xp_des, T] = MinJerkTrajectory.plan_plate_trajectory(Tb, x_p0, x_pTb, ub_0, -ub_0, ap_0, ap_T);

    % 2. Compute optimal input velocities u_des
    [u_des] = compute_optimal_input_signal(xp_des, dup);

    % 3. Simulate the calculated inputs
    [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec] = simulate_one_iteration(dt, N, x_b0, x_p0, u_b0, u_b0, u, false);

    % 4.Identify errors d1, d2, dp_j (kalman filter or optimization problem
    [d1, d2, dup] = filter_disturbances(x_b, u_b, x_p, u_p, d1, d2, dup);
end


%% Components
% 1. Ball Trajectory
function [Tb, ub_0] = plan_ball_trajectory(hb, d1, d2)
    global g;
    ub_0 = sqrt(2*g*(hb - d1));  % velocity of ball at throw point
    Tb = 2*ub_0/g + d2; % flying time of the ball
end

% 3. Calculate Optimal Input Signal
function [u_des] = compute_optimal_input_signal(xp_des, dup)
% xp_des
% dup

    u_des = [];
end

% 4. Calculate Optimal Input Signal
function [d1, d2, dup] = filter_disturbances(x_b, u_b, x_p, u_p, d1, d2, dup)
    d1 = [];
    d2 = [];
    dup = [];
end