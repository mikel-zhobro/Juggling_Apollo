%% Parameters
% Constants
g = 9.80665;    % [m/s^2]
dt = 0.004;      % [s] discretization time step size

% Params
m_b = 0.03;     % [kg]
m_p = 10;      % [kg]
k_c = 10;       % [1/s]  time-constant of velocity controller

%% Feedforward controled system
% Here we want to set some convention to avoid missunderstandins later on.
% 1. the state is [xb, xp, ub, up]^T
% 2. the system can have as input either velocity u_des or the force F_p
% 2.

% Design params
h_b_max = 1;                % [m] maximal height the ball achievs
input_is_force = false;

% SYSTEM DYNAMICS
sys = DynamicSystem('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'dt', dt);
if input_is_force
    [Ad, Bd, Cd, S, c] = sys.getSystemMarixesForceControl(dt, false);
    [Ad_impact, Bd_impact, ~, ~, c_impact] = sys.getSystemMarixesForceControl(dt, true);

else
    [Ad, Bd, Cd, S, c] = sys.getSystemMarixesVelocityControl(dt, false);
    [Ad_impact, Bd_impact, ~, ~, c_impact] = sys.getSystemMarixesVelocityControl(dt, true);
end

% SIMULATION
sim = Simulation('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'input_is_force', input_is_force, 'sys', sys);

% INIT throw and catch point
[Tb, ub_0] = plan_ball_trajectory(h_b_max, 0, 0);   % [s] flying time of the ball
x_b0 = 0;       x_p0 = x_b0;    up_0 = ub_0;   ap_0 = -3*g;
x_pTb = x_p0;                                  ap_T = 0;
x0 = [x_b0; x_p0; ub_0; up_0];

% DESIRED INPUT OPTIMIZER
desired_input_optimizer = OptimizationDesiredInput('Ad', Ad, 'Bd', Bd,          ...
                                                   'Cd', Cd, 'S', S,            ...
                                                   'x0', x0, 'c', c,            ...
                                                   'Ad_impact', Ad_impact,      ...
                                                   'Bd_impact', Bd_impact,      ...
                                                   'c_impact', c_impact         );
% KALMAN FILTER
% Iteration params
T = 2 * Tb;                                         % [s] time for one iteration T = 2 T_b
N = Simulation.steps_from_time(T, dt);              % number of steps for one iteration
% Sizes
n_x = length(x0);
n_y = size(Cd, 1);
n_dup = 2;
% Init covariances
P0 = 2*eye(n_dup*N, n_dup*N); % initial disturbance covariance
M = 0.1 * eye(n_y*N, n_y*N);
epsilon = 0.3;
% Init disturbances
d1 = 0;
d2 = 0;
dup = zeros(n_dup*N,1);
% KF
ilc_kf = ILCKalmanFilter('dt', dt, 'M', M, 'd0', dup, 'P0', P0, 'epsilon0', epsilon);

%% Iteration
ILC_it = 12; % number of ILC iteration

% collect: dup, x_p, x_b, u_p
dup_vec = zeros(ILC_it, N, n_dup);
x_p_vec = zeros(ILC_it, N);
x_b_vec = zeros(ILC_it, N);
u_p_vec = zeros(ILC_it, N);

% PROBLEM: it diverges after iteration 10
ilc_kf.resetKF()
for j = 1:ILC_it
%     close all
    display("ITERATION: " + num2str(j))
tic
    % 0. Compute Ball Height and Time
    [Tb, ub_0] = plan_ball_trajectory(h_b_max, d1, d2);
    ub_T = -ub_0/4;

    % 1. Plan desired Plate trajectory (min jerk trajectory)
%     [xuaj_des] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_T, ub_0);              % free start and end acceleration
%     [xuaj_des] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, ub_T, ap_0);        % free end acceleration
    [xuaj_des] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, ub_T, ap_0, ap_T);    % set start and end acceleration
%     MinJerkTrajectory.plot_paths(xuaj_des, dt, 'Planned')
    Cd = [1 0 0 0; ...
          0 1 0 0];
    r = Cd*xuaj_des;


    % 2. Compute optimal input signal
    set_of_impact_timesteps = ones(1, Simulation.steps_from_time(T, dt));
    set_of_impact_timesteps(1:2) = 2;
    set_of_impact_timesteps(Simulation.steps_from_time(Tb, dt):end) = 2;
    [u_des] = desired_input_optimizer.calcDesiredInput(dup, r(:), set_of_impact_timesteps); % updates the used lifted space matrixes G, F etc..

    % 3. Simulate the calculated inputs
    repetitions = 1;
    [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, u_vec] = sim.simulate_one_iteration(dt, T, x_b0, x_p0, ub_0, ub_0, u_des, repetitions);

    % Plotting for Simulation Example
    % Simulation.plot_results(dt, u_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
    intervals = (Simulation.find_continuous_intervals(find(gN_vec<=1e-5)) -1) *dt
    intervals_expected = (Simulation.find_continuous_intervals(find(set_of_impact_timesteps==2)) -1) *dt

    % 4.Identify errors d1, d2, dup (kalman filter or optimization problem)
    % dup = [x_p; u_p];
    y = [x_p; u_p]; y = y(:);
    % PROBLEM: ball position and velocity disturbances are not estimated
    % so dup should track only x_p and u_p
    ilc_kf.set_G_GF_Gd0(desired_input_optimizer.G, desired_input_optimizer.GF, desired_input_optimizer.Gd0);
    dup = ilc_kf.updateStep2(u_des, y);
toc

    % collect data for plotting
    for tt=1:n_dup
        dup_vec(j,:,tt) =  dup(tt:n_dup:end);
    end
    x_p_vec(j,:) =  x_p;
    x_b_vec(j,:) =  x_b;
    u_p_vec(j,:) =  u_p;
end


%% Plot trajectories over ILC iterations
close all
Simulation.plot_results(dt, u_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
% names = ["x_b", "x_p", "u_b", "u_p"];
% for tt=[2,4]
names = ["x_p", "u_p"];
for tt=[1,2]
    plotIterations(squeeze(dup_vec(:,:,tt)) - squeeze(dup_vec(1,:,tt)), "d_" +names(tt)+ "through iterations")% - squeeze(dup_vec(1,:,tt))
end
plotIterations(x_b_vec, "x_b through iterations", dt)
plotIterations(x_p_vec, "x_p through iterations", dt)
plotIterations(u_p_vec, "u_p through iterations", dt)

%% Components
% Ball Trajectory
% PROBLEM: does not consider impuls
function [Tb, ub_0] = plan_ball_trajectory(hb, d1, d2, mp, mb, dt, Fp)
    g = 9.80665;    % [m/s^2]
    ub_0 = sqrt(2*g*(hb - d1));  % velocity of ball at throw point
    Tb = 2*ub_0/g + d2; % flying time of the ball
    if nargin > 3 % consider impuls
        dPN = mb*mp/(mb+mp) * (g*dt +Fp*dt/mp);
        ub_0 = ub_0 - dPN;
    end
end

function plotIterations(y, tititle, dt)
    if nargin<3
        dt = 1;
    end

    n_x = size(y,2);
    n_i = size(y,1);
    
    legend_vec = arrayfun(@(i)("iteration "+ num2str(i)),(1:n_i));
    figure
    hold on
    timesteps = (1:n_x)*dt;
    plot(timesteps, y)
    legend(legend_vec)
    title(tititle)

end