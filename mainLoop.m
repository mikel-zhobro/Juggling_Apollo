%% Parameters
% Constants
g = 9.80665;    % [m/s^2]
dt = 0.004;      % [s] discretization time step size

% Params
m_b = 0.1;     % [kg]
m_p = 10;      % [kg]
k_c = 10;       % [1/s]  time-constant of velocity controller

%% Feedforward controled system
% Here we want to set some convention to avoid missunderstandins later on.
% 1. the state is [xb, xp, ub, up]^T
% 2. the system can have as input either velocity u_des or the force F_p

% Design params
h_b_max = 1;                % [m] maximal height the ball achievs
input_is_force = false;

% INIT
% ----------------------------------------------------------------------- %
% Throw and catch point
[Tb, ub_0] = plan_ball_trajectory(h_b_max, 0, 0);   % flying time of the ball and required init velocity
x_b0 = 0;       x_p0 = x_b0;    up_0 = ub_0;        ap_0 = -3*g;
x_pTb = x_p0;                   ub_T = -ub_0/6;     ap_T = 0;
                              %----important----%
x0 = [x_b0; x_p0; ub_0; up_0];

% Iteration params
T = 2 * Tb;                                         % [s] time for one iteration T = 2 T_b
N = Simulation.steps_from_time(T, dt);              % number of steps for one iteration (1 more than size of input and states we can change)
% ----------------------------------------------------------------------- %

% I. MIN-JERK TRAJECTORY
    [xuaj_des] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, ub_T);              % free start and end acceleration
%     [xuaj_des] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, ub_T, ap_0);        % free end acceleration
% [xuaj_des] = MinJerkTrajectory.plan_plate_trajectory(dt, Tb, x_p0, x_pTb, ub_0, ub_T, ap_0, ap_T);    % set start and end acceleration
MinJerkTrajectory.plot_paths(xuaj_des, dt, "free start and end acceleration")
y_des = transpose(xuaj_des(1,2:end));


% II. SYSTEM DYNAMICS
sys = DynamicSystem('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'dt', dt);
if input_is_force
    [Ad, Bd, Cd, S, c] = sys.getSystemMarixesForceControl(dt, false);
    [Ad_impact, Bd_impact, ~, ~, c_impact] = sys.getSystemMarixesForceControl(dt, true);

else
    [Ad, Bd, Cd, S, c] = sys.getSystemMarixesVelocityControl(dt, false);
    [Ad_impact, Bd_impact, ~, ~, c_impact] = sys.getSystemMarixesVelocityControl(dt, true);
end
% Sizes
n_x = size(Ad, 1);
n_y = size(Cd, 1);
n_dup = size(S, 2);
N_1 = N -1;  % important since input cannot influence the first state
% III. LIFTED STATE SPACE
lifted_state_space = LiftedStateSpace('Ad', Ad, 'Bd', Bd,        ...
                                      'Cd', Cd, 'S', S,          ...
                                      'x0', x0, 'c', c,          ...
                                      'Ad_impact', Ad_impact,    ...
                                      'Bd_impact', Bd_impact,    ...
                                      'c_impact', c_impact       );

% IV. SIMULATION
sim = Simulation('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'input_is_force', input_is_force, 'sys', sys, 'air_Drag', true);

% V. DESIRED INPUT OPTIMIZER
desired_input_optimizer = OptimizationDesiredInput(lifted_state_space);

% VI. KALMAN FILTER
% Init disturbances
d1 = 0; d2 = 0; dup = zeros(n_dup*N_1,1);
ilc_kf = ILCKalmanFilter('lss', lifted_state_space,             ...
                         'M' , 0.1 * eye(n_y*N_1, n_y*N_1),         ...
                         'P0', 1e-2*eye(n_dup*N_1, n_dup*N_1),      ...
                         'dt', dt, 'd0', dup, 'epsilon0', 0.3   );

%% Iteration
ILC_it = 1; % number of ILC iteration

% collect: dup, x_p, x_b, u_p
dup_vec = zeros(ILC_it, n_dup*N_1, n_dup);
x_p_vec = zeros(ILC_it, N);
x_b_vec = zeros(ILC_it, N);
u_p_vec = zeros(ILC_it, N);
u_des_vec = zeros(ILC_it, N_1);

% ILC
ilc_kf.resetKF()
close all

% Initialize lifted state space
impact_timesteps = ones(1, N_1);
impact_timesteps(1) = 2;
impact_timesteps(Simulation.steps_from_time(Tb, dt):end) = 2;
lifted_state_space.updateQuadrProgMatrixes(impact_timesteps)
%% ILC Loop
MinJerkTrajectory.plot_paths(xuaj_des, dt, "free start and end acceleration")
for j = 1:ILC_it
    display("ITERATION: " + num2str(j))

    % 1. Compute optimal input signal
    u_des = desired_input_optimizer.calcDesiredInput(dup, y_des); % updates the used lifted space matrixes G, F etc..

    % 2. Simulate the calculated inputs
    disp("SIMULATION")
    repetitions = 1;
    [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = sim.simulate_one_iteration(dt, T, x_b0, x_p0, ub_0, ub_0, u_des, repetitions);


    % 3. Find intervals of impact and update impact_timesteps and update
    % lifted state space matrixes accordingly so KF usees the right
    % model.(not sure about the input optimizer!!)
    disp("Interval comparison")
    intervals = Simulation.find_continuous_intervals(find(gN_vec(2:end)<=1e-5));
    intervals_expected = Simulation.find_continuous_intervals(find(impact_timesteps==2));
    disp("real intervals"); disp((intervals)*dt);
    disp("expected intervals"); disp((intervals_expected)*dt); % no -1 to the interval since we have considered it above

% tic % takes a lot of time
%     disp("Update Lifted Space")
%     impact_timesteps(:) = 1; % reset to 1
%     for i_n=1:size(intervals,2)
%         i_val = intervals(:,i_n);
%         start = i_val(1);
%         endd = i_val(2);
%         impact_timesteps(start:end) = 2;
%     end
%     lifted_state_space.updateQuadrProgMatrixes(impact_timesteps); % updates automatically the matrixes in ILC and input optimizer
% toc

    % 4.Identify errors d1, d2, dup (kalman filter or optimization problem)
    disp("KF update step")
    dup = ilc_kf.updateStep(u_des, transpose(x_p(2:end)));

    % collect data for plotting
    dup_vec(j,:) =  dup;
    x_p_vec(j,:) =  x_p;
    x_b_vec(j,:) =  x_b;
    u_p_vec(j,:) =  u_p;
    u_des_vec(j,:) =  u_des;
end
Simulation.plot_results(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)

%% Plot trajectories over ILC iterations
close all
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
