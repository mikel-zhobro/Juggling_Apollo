%% Parameters
% Constants
clear all
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
[Tb, ub_00] = plan_ball_trajectory(h_b_max, 0, 0);   % flying time of the ball and required init velocity

% Init State
ub_throw = ub_00;
x_b0 = 0;       ub_0 = ub_throw;
x_p0 = 0;       up_0 = ub_throw;         % acc.ta = -3*g;
x_pTb = 0;      up_T = -ub_throw/6;       % acc.tb = 0;
              %----important----%

x0 = {x_b0; x_p0; ub_0; up_0}; % ball about to be thrown

% Minjerk
[y_des, vp, ap, jp] = MinJerkTrajectory2.get_min_jerk_trajectory(dt, 0, Tb, x_p0, x_pTb, up_0, up_T);
MinJerkTrajectory2.plot_paths(y_des, vp, ap, jp, dt, 'eha')

% Kalman Filter Params
T_plate_dist = 0.5*dt/m_p; % for the case we model dp as the disturbance on dPN(is little weird)
T_plate_dist =1;
T_dist_plate = 1/T_plate_dist;

kf_dpn_params.P0_diag = T_dist_plate^2 * 0.002;       % the diagonal value of initial variance of disturbance
kf_dpn_params.epsilon0 = T_dist_plate^2 * 0.001;      % initial variance of noise on the disturbance
kf_dpn_params.epsilon_decrease_rate = 0.9;          % the decreasing factor of noise on the disturbance
kf_dpn_params.M_diag = 0.1;                         % diagonal value of covarianc of noise on the measurment

kf_d1d2_params.P0_diag = 0.1;
kf_d1d2_params.epsilon0 = 0.1;
kf_d1d2_params.epsilon_decrease_rate = 0.9;
kf_d1d2_params.M_diag = 0.1;

% ILC
my_ilc = ILC('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'dt', dt,              ...
             'x_0', cell2mat(x0), 't_f', Tb, 'input_is_force', input_is_force,  ...
             'kf_d1d2_params', kf_d1d2_params, 'kf_dpn_params', kf_dpn_params)  ;
y_des = y_des(2:end);
[u_ff] = my_ilc.learnPlateMotionStep(y_des); %% resets impact_timesteps for the liftest state space

% Iteration params
sim = Simulation('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'input_is_force', input_is_force, 'sys', my_ilc.sys, 'air_drag', false);
% ----------------------------------------------------------------------- %

%% Learn Plate Motion
% close all
ILC_it = 45; % number of ILC iteration

% reset ilc
my_ilc.resetILC() %% resets the kalman filters

% collect: dup, x_p, x_b, u_p
dup_vec = zeros([ILC_it, size(my_ilc.kf_dpn.d)]);
x_p_vec = zeros(ILC_it, my_ilc.N_1 + 1);
x_b_vec = zeros(ILC_it, my_ilc.N_1 + 1);
u_p_vec = zeros(ILC_it, my_ilc.N_1 + 1);
u_des_vec = zeros(ILC_it, my_ilc.N_1);

% ILC Loop
% disturbance to be learned
period = 0.1/dt;
disturbance = 200*sin(2*pi/period*(0:my_ilc.N_1-1)); % disturbance on the plate position
for j = 1:ILC_it
    display("ITERATION: " + num2str(j))

    % Main Simulation
    [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = sim.simulate_one_iteration(dt, my_ilc.t_f, x0{:}, u_ff, 1, disturbance);

    % Collect data for plotting
    dup_vec(j,:) =  my_ilc.kf_dpn.d;
    x_p_vec(j,:) =  x_p;
    x_b_vec(j,:) =  x_b;
    u_p_vec(j,:) =  u_p;
    u_des_vec(j,:) =  F_vec(1:end-1);

    % LEARN Platte Motion
    %                                                   from 2 since we cant optimize the first state x(0)
    [u_ff] = my_ilc.learnPlateMotionStep(y_des, u_ff, transpose(x_p(2:end)));
%     Simulation.plot_results(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
end

Simulation.plot_results(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
% plotIterations(T_plate_dist*dup_vec - disturbance, "d_{P_N} - d_{real} through iterations", dt, 1)
%% Plot trajectories over ILC iterations
close all
% plotIterations(T_plate_dist*dup_vec, "d_{P} through iterations", dt, 1)
% plot((0:my_ilc.N_1-1)*dt, disturbance, 'x');
% plotIterations(x_b_vec, "x_b through iterations", dt, 1)
plotIterations([x_p_vec(end,2:end)- y_des; dup_vec(end,:)], "x_p through iterations", dt, 1)
% plotIterations(u_p_vec, "u_p through iterations", dt, 4)
% plotIterations(u_des_vec, "u_{ff} through iterations", dt, 3)

%%






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
    timesteps = (0:n_x-1)*dt;
    plot(timesteps, y(1:every_n:end,:))
    legend(legend_vec)
    title(tititle)

end
