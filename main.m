
%% Parameters
% Constants
g = 9.80665;    % [m/s^2]
dt = 0.01;      % [s] discretization time step size

% Params
m_b = 0.03;     % [kg]
m_p = 100;      % [kg]
k_c = 10;       % [1/s]  time-constant of velocity controller

%% Simulation Example
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
F_p = 100 * m_p * A*sin(pi/Tb *timesteps);        % [N] input force on the plate

% Simulation for N steps
sim = Simulation('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'dt', dt);
[x_b, u_b, x_p, u_p, dP_N_vec, gN_vec] = sim.simulate_one_iteration(dt, Tsim, x_b0, x_p0, u_b0, u_p0, F_p);

% Plotting for Simulation Example
close all
Simulation.plot_results(dt, F_p, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)

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
myopt = OptimizationDesiredInput('Ad', Ad, 'Bd', Bd, ...
                                 'Cd', Cd, 'S', S,   ...
                                 'x0', x0, 'c', c, 'N', N);
% Test solving the quadratic problem
dup = zeros(N,1);
y_des = ones(2*N,1);
u_des = myopt.calcDesiredInput(dup, y_des);
display(u_des);

%% Feedforward controled system
% Design params
h_b_max = 1;                % [m] maximal height the ball achievs
M = 10;                     % number of ILC iteration
force_conrolled = false;
% Initialize disturbances
d1 = 0;
d2 = 0;
dup = 0;

% Initialize throw point
x_b0 = 0;
x_p0 = x_b0;
x_pTb = x_p0;


% Set up simulation
[Tb, ~] = plan_ball_trajectory(h_b_max, d1, d2);    % [s] flying time of the ball
T = 2 * Tb;                                         % [s] time for one iteration T = 2 T_b
N = Simulation.steps_from_time(T, dt);              % number of steps for one iteration (mayve use floor)
sim = Simulation('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g);

% Set up desired input optimizer
sys = DynamicSystem('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'dt', dt);
if force_controlled
    [Ad, Bd, Cd, S, c] = getSystemMarixesForceControl(dt);
else
    [Ad, Bd, Cd, S, c] = sys.getSystemMarixesVeocityControl(dt);
end
x0 = [0;0;0;0]; % [xb0; xp0; ub0; up0]
desired_input_optimizer = OptimizationDesiredInput('Ad', Ad, 'Bd', Bd, ...
                                                   'Cd', Cd, 'S', S,   ...
                                                   'x0', x0, 'c', c, 'N', N);
for j = j:M
    % 0. Compute Ball Height and Time
    [Tb, ub_0] = plan_ball_trajectory(h_b_max, d1, d2);
    
    % 1. Plan desired Plate trajectory (min jerk trajectory)
    [xp_des, T] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, -ub_0);                  % free start and end acceleration
%     [xp_des, T] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, -ub_0, ap_0);          % free end acceleration
%     [xp_des, T] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, -ub_0, ap_0, ap_T);    % set start and end acceleration

    % 2. Compute optimal input signal
    [u_des] = desired_input_optimizer.calcDesiredInput(dup, y_des);

    % 3. Simulate the calculated inputs
    [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec] = sim.simulate_one_iteration(dt, T, x_b0, x_p0, u_b0, u_b0, u, force_conrolled);

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
    d1 = [];
    d2 = [];
    dup = [];
end