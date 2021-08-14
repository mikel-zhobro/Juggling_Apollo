
%% Parameters
% Constants
g = 9.80665;    % [m/s^2]
dt = 0.01;      % [s] discretization time step size

% Params
m_b = 0.03;     % [kg]
m_p = 10;      % [kg]
k_c = 10;       % [1/s]  time-constant of velocity controller

%% Simulation Example
close all
% Initial Conditions
x_b0 = 0;       % [m]   starting ball position
x_p0 = 0;       % [m]   starting plate position
u_p0 = 0;       % [m/s] starting plate velocity
u_b0 = u_p0;    % [m/s] starting ball velocity

% Design params
h_b_max = 1;                                        % [m] maximal height the ball achievs
[Tb, ~] = plan_ball_trajectory(h_b_max, 0, 0);      % [s] flying time of the ball
Tsim= Tb*2*5;                                       % [s] simulation time
N = Simulation.steps_from_time(Tsim, dt);           % number of steps for one iteration (maybe use floor)

% Input
A = 0.3;                                            % [m] amplitude
timesteps = dt * (0:N);                             % [s,s,..] timesteps
F_p = 100 * m_p * A*sin(pi/Tb *timesteps);          % [N] input force on the plate
input_is_force = false;
% Simulation for N steps
sys = DynamicSystem('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'dt', dt);
sim = Simulation('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'input_is_force', input_is_force, 'sys', sys);
[x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = sim.simulate_one_iteration(dt, Tsim, x_b0, x_p0, u_b0, u_p0, F_p);

[x_b2, u_b2, x_p2, u_p2, dP_N_vec2, gN_vec2, F_vec2] = sim.simulate_one_iteration2(dt, Tsim, x_b0, x_p0, u_b0, u_p0, F_p);
% Plotting for Simulation Example
% close all
Simulation.plot_results(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)

Simulation.plot_results(dt, F_vec2, x_b2, u_b2, x_p2, u_p2, dP_N_vec2, gN_vec2)
Simulation.plot_results(dt, F_vec-F_vec2, x_b-x_b2, u_b-u_b2, x_p-x_p2, u_p-u_p2, dP_N_vec-dP_N_vec2, gN_vec-gN_vec2)

display(["norm(x_b-x_b2): ", norm(x_b-x_b2)]);
display(["norm(u_b-u_b2): ", norm(u_b-u_b2)]);
display(["norm(x_p-x_p2): ", norm(x_p-x_p2)]);
display(["norm(u_p-u_p2): ", norm(u_p-u_p2)]);
display(["norm(dP_N_vec-dP_N_vec2): ", norm(dP_N_vec-dP_N_vec2)]);
display(["norm(gN_vec-gN_vec2): ", norm(gN_vec-gN_vec2)]);
display(["norm(F_vec-F_vec2): ", norm(F_vec-F_vec2)]);
%% Desired Trajectory planning example
% Initialize disturbances
d1 = 0;         % disturbance1
d2 = 0;         % disturbance2

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
    [xp_des, T] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, -ub_0);
    MinJerkTrajectory.plot_paths(xp_des, T, dt, 'Free start and end acceleration')

    % 2) Free end acceleration
    [xp_des, T] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, -ub_0, ap_0);
    MinJerkTrajectory.plot_paths(xp_des, T, dt, 'Free end acceleration')

    % 3) Set start and end acceleration
    [xp_des, T] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, -ub_0, ap_0, ap_T);
    MinJerkTrajectory.plot_paths(xp_des, T, dt, 'Set start and end acceleration')

%% Desired Input optimization example
% Create Ad, Bd, Cd, S, c and x0
N = 4;
Ad = diag(1:3);
Bd = [2;2;2];
Cd = ones(2,3); Cd(1,1) = 0;
S = ones(3,1);
x0 = [0.1;0.1;0.1];
c = [0.1;0.1;0.1];

% Test initialization of matrixes
myopt = OptimizationDesiredInput('Ad', Ad, 'Bd', Bd,        ...
                                 'Cd', Cd, 'S', S,          ...
                                 'x0', x0, 'c', c,          ...
                                 'Ad_impact', Ad,           ...
                                 'Bd_impact', Bd,           ...
                                 'c_impact', c              );
% Test solving the quadratic problem
dup = zeros(N,1);
y_des = ones(2*N,1);
set_of_impact_timesteps = ones(1,N);

u_des = myopt.calcDesiredInput(dup, y_des, set_of_impact_timesteps);
display(u_des);

%% Feedforward controled system
% Here we want to set some convention to avoid missunderstandins later on.
% 1. the state is [xb, xp, ub, up]^T
% 2. the system can have as input either velocity u_des or the force F_p
% 2.

% Design params
h_b_max = 1;                % [m] maximal height the ball achievs
M = 1;                      % number of ILC iteration
input_is_force = true;

% Set up state space matrixes
sys = DynamicSystem('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'dt', dt);
if input_is_force
    [Ad, Bd, Cd, S, c] = sys.getSystemMarixesForceControl(dt, false);
    [Ad_impact, Bd_impact, ~, ~, c_impact] = sys.getSystemMarixesForceControl(dt, true);
else
    [Ad, Bd, Cd, S, c] = sys.getSystemMarixesVelocityControl(dt, false);
    [Ad_impact, Bd_impact, ~, ~, c_impact] = sys.getSystemMarixesVelocityControl(dt, true);
end

% Set up simulation
[Tb, ub_0] = plan_ball_trajectory(h_b_max, 0, 0);   % [s] flying time of the ball
T = 2 * Tb;                                         % [s] time for one iteration T = 2 T_b
N = Simulation.steps_from_time(T, dt);              % number of steps for one iteration
sim = Simulation('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'input_is_force', input_is_force, 'sys', sys);

% Initialize throw point
x_b0 = 0;
x_p0 = x_b0;
x_pTb = x_p0;
ap_0 = 0;
ap_T=0;
up_0 = ub_0;

% Initialize disturbances
d1 = 0;
d2 = 0;
dup = zeros(N,1);

% Set up desired input optimizer
x0 = [x_b0; x_p0; ub_0; up_0];
desired_input_optimizer = OptimizationDesiredInput('Ad', Ad, 'Bd', Bd,          ...
                                                   'Cd', Cd, 'S', S,            ...
                                                   'x0', x0, 'c', c,            ...
                                                   'Ad_impact', Ad_impact,      ...
                                                   'Bd_impact', Bd_impact,      ...
                                                   'c_impact', c_impact         );
for j = 1:M
    close all
    % 0. Compute Ball Height and Time
    [Tb, ub_0] = plan_ball_trajectory(h_b_max, d1, d2);

    % 1. Plan desired Plate trajectory (min jerk trajectory)
    % [xuaj_des, T] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, -ub_0);              % free start and end acceleration
    % [xuaj_des, T] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, -ub_0, ap_0);        % free end acceleration
    [xuaj_des, T] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, -ub_0, ap_0, ap_T);    % set start and end acceleration
%     MinJerkTrajectory.plot_paths(xuaj_des, T, dt, 'Planned')

    % 2. Compute optimal input signal
    set_of_impact_timesteps = ones(1, Simulation.steps_from_time(T, dt));
    set_of_impact_timesteps(1) = 2;
    set_of_impact_timesteps(Simulation.steps_from_time(Tb, dt):end) = 2;
    [u_des] = desired_input_optimizer.calcDesiredInput(dup, transpose(xuaj_des(1,:)), set_of_impact_timesteps);

    % 3. Simulate the calculated inputs
    [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, u_vec] = sim.simulate_one_iteration2(dt, T, x_b0, x_p0, ub_0, ub_0, u_des);

    % Plotting for Simulation Example
    Simulation.plot_results(dt, u_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)

    % 4.Identify errors d1, d2, dp_j (kalman filter or optimization problem
    [d1, d2, dup] = filter_disturbances(x_b, u_b, x_p, u_p, d1, d2, dup);
end


%% Components
% 1. Ball Trajectory
function [Tb, ub_0] = plan_ball_trajectory(hb, d1, d2)
    g = 9.80665;    % [m/s^2]
    ub_0 = sqrt(2*g*(hb - d1));  % velocity of ball at throw point
    Tb = 2*ub_0/g + d2; % flying time of the ball
end

% 4. Estimate disturbances
function [d1, d2, dup] = filter_disturbances(x_b, u_b, x_p, u_p, d1, d2, dup)
%     d1 = d1;
%     d2 = d2;
%     dup = dup;
end
