%% Parameters
% Constants
g = 9.80665;    % [m/s^2]
dt = 0.004;      % [s] discretization time step size

% Params
m_b = 1;     % [kg]
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
n_dup = 1;
% Init covariances
P0 = 1e-2*eye(n_dup*N, n_dup*N); % initial disturbance covariance
M = 0.1 * eye(n_y*N, n_y*N);
epsilon = 0.3;
% Init disturbances
d1 = 0;
d2 = 0;
dup = zeros(n_dup*N,1);
% KF
ilc_kf = ILCKalmanFilter('dt', dt, 'M', M, 'd0', dup, 'P0', P0, 'epsilon0', epsilon);

%% Iteration
ILC_it = 2; % number of ILC iteration

% collect: dup, x_p, x_b, u_p
dup_vec = zeros(ILC_it, N, n_dup);
x_p_vec = zeros(ILC_it, N);
x_b_vec = zeros(ILC_it, N);
u_p_vec = zeros(ILC_it, N);
u_des_vec = zeros(ILC_it, N);

% ILC
ilc_kf.resetKF()
close all

% 0. Compute Ball Height and Time
[Tb, ub_0] = plan_ball_trajectory(h_b_max, d1, d2);
ub_T = -ub_0/6;
% 1. Plan desired Plate trajectory (min jerk trajectory)
%     [xuaj_des] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_T, ub_0);              % free start and end acceleration
%     [xuaj_des] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, ub_T, ap_0);        % free end acceleration
[xuaj_des] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, ub_T, ap_0, ap_T);    % set start and end acceleration
%     MinJerkTrajectory.plot_paths(xuaj_des, dt, 'Planned')
%     Cd = [1 0 0 0; ...
%           0 1 0 0];
%     r = Cd*xuaj_des;
r = xuaj_des(1,:);

% Initialize desired_input_optimizer
set_of_impact_timesteps = ones(1, Simulation.steps_from_time(T, dt));
set_of_impact_timesteps(1:2) = 2;
set_of_impact_timesteps(Simulation.steps_from_time(Tb, dt):end) = 2;
desired_input_optimizer.updateQuadrProgMatrixes(set_of_impact_timesteps)
% ILC Loop
for j = 1:ILC_it
    display("ITERATION: " + num2str(j))
tic
    % 2. Compute optimal input signal
    [u_des] = desired_input_optimizer.calcDesiredInput(dup, r(:), set_of_impact_timesteps); % updates the used lifted space matrixes G, F etc..

    % 3. Simulate the calculated inputs
    repetitions = 1;
    [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, u_vec] = sim.simulate_one_iteration(dt, T, x_b0, x_p0, ub_0, ub_0, u_des, repetitions);

    % Plotting for Simulation Example
    % Simulation.plot_results(dt, u_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
    intervals = Simulation.find_continuous_intervals(find(gN_vec<=1e-5));
    disp("real intervals"); disp((intervals-1)*dt);
    intervals_expected = Simulation.find_continuous_intervals(find(set_of_impact_timesteps==2));
    disp("expected intervals"); disp((intervals_expected-1)*dt);
    % update trajectory(lifted space)matrixes according to impact intervals
    set_of_impact_timesteps(:) = 1;
    for i_n=1:size(intervals,2)
        i_val = intervals(:,i_n);
        start = i_val(1);
        endd = i_val(2);
        set_of_impact_timesteps(start:end) = 2;
    end
    desired_input_optimizer.updateQuadrProgMatrixes(set_of_impact_timesteps)

    % 4.Identify errors d1, d2, dup (kalman filter or optimization problem)
    % dup = [x_p; u_p];
%     y = [x_p; u_p]; y = y(:);
      y = transpose(x_p);
    % PROBLEM: ball position and velocity disturbances are not estimated
    % so dup should track only x_p and u_p
    ilc_kf.set_G_GF_Gd0(desired_input_optimizer.G, desired_input_optimizer.GF, desired_input_optimizer.Gd0,  desired_input_optimizer.GK);
    dup = ilc_kf.updateStep3(u_des, y);
toc

    % collect data for plotting
%     for tt=1:n_dup
%         dup_vec(j,:,tt) =  dup(tt:n_dup:end);
%     end
    dup_vec(j,:) =  dup;
    x_p_vec(j,:) =  x_p;
    x_b_vec(j,:) =  x_b;
    u_p_vec(j,:) =  u_p;
    u_des_vec(j,:) =  u_des;
end
Simulation.plot_results(dt, u_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)

%% Plot trajectories over ILC iterations
close all
% names = ["x_p", "u_p"];
% for tt=[1,2]
%     plotIterations(squeeze(dup_vec(:,:,tt)) - squeeze(dup_vec(1,:,tt)), "d_" +names(tt)+ "through iterations")% - squeeze(dup_vec(1,:,tt))
% end
plotIterations(dup_vec, "d_{P_N} through iterations", dt, 4)
% plotIterations(x_b_vec, "x_b through iterations", dt, 4)
% plotIterations(x_p_vec, "x_p through iterations", dt, 4)
% plotIterations(u_p_vec, "u_p through iterations", dt, 4)
plotIterations(u_des_vec - u_des_vec(1,:), "u_{des} through iterations", dt, 4)
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

function plotIterations(y, tititle, dt, every_n)
    if nargin<3
        dt = 1;
    end    
    if nargin<4
        every_n = 1;
    end

    n_x = size(y,2);
    n_i = size(y,1);
    
    legend_vec = arrayfun(@(i)("iteration "+ num2str(i)),(1:every_n:n_i));
    figure
    hold on
    timesteps = (1:n_x)*dt;
    plot(timesteps, y(1:every_n:end,:))
    legend(legend_vec)
    title(tititle)

end