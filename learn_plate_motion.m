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
x_ruhe = -0.4;
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
x_pTb = 0;      ub_T = -ub_throw/6;       % acc.tb = 0;
              %----important----%

x0 = {x_b0; x_p0; ub_0; up_0}; % ball about to be thrown

% Minjerk
[y_des, vp, ap, jp] = MinJerkTrajectory2.get_min_jerk_trajectory(dt, 0, Tb, x_p0, x_pTb, up_0, ub_T);
MinJerkTrajectory2.plot_paths(y_des, vp, ap, jp, dt, 'eha')

% Kalman Filter Params
T_plate_dist = 0.5*dt/m_p;
T_dist_plate = 1/T_plate_dist;

kf_dpn_params.P0_diag = T_dist_plate^2 * 0.2;       % the diagonal value of initial variance of disturbance
kf_dpn_params.epsilon0 = T_dist_plate^2 * 0.3;      % initial variance of noise on the disturbance
kf_dpn_params.epsilon_decrease_rate = 0.9;          % the decreasing factor of noise on the disturbance
kf_dpn_params.M_diag = 0.1;                         % diagonal value of covarianc of noise on the measurment

kf_d1d2_params.P0_diag = 0.1;
kf_d1d2_params.epsilon0 = 0.1;
kf_d1d2_params.epsilon_decrease_rate = 0.9;
kf_d1d2_params.M_diag = 0.1;                     

% ILC
my_ilc = ILC('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'dt', dt,              ...
             'x_ruhe', x_ruhe, 't_f', Tb, 'ub_0', ub_00,                         ...
             'kf_d1d2_params', kf_d1d2_params, 'kf_dpn_params', kf_dpn_params)  ;
y_des = y_des(2:end);
[u_ff] = my_ilc.learnPlateMotionStep(y_des); %% resets impact_timesteps for the liftest state space

% Iteration params
sim = Simulation('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'input_is_force', input_is_force, 'sys', my_ilc.sys, 'air_drag', false);
% ----------------------------------------------------------------------- %

%% Learn Plate Motion
ILC_it = 1; % number of ILC iteration

% reset ilc
% my_ilc.resetILC() %% resets the kalman filters

% collect: dup, x_p, x_b, u_p
dup_vec = zeros([ILC_it, size(my_ilc.kf_dpn.d)]);
x_p_vec = zeros(ILC_it, my_ilc.N_1 + 1);
x_b_vec = zeros(ILC_it, my_ilc.N_1 + 1);
u_p_vec = zeros(ILC_it, my_ilc.N_1 + 1);
u_des_vec = zeros(ILC_it, my_ilc.N_1);

% ILC Loop
% disturbance to be learned
period = 0.3/dt;
disturbance = 0*sin(2*pi/period*[0:Simulation.steps_from_time(my_ilc.t_f, dt)-2]);
for j = 1:ILC_it
    display("ITERATION: " + num2str(j))
    
    % Main Simulation
    [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = sim.simulate_one_iteration_ss(dt, my_ilc.t_f, x0{:}, u_ff, 1, disturbance);
    
    % LEARN Platte Motion
    %                                                   from 2 since we cant optimize the first state x(0)
    [u_ff] = my_ilc.learnPlateMotionStep(y_des, u_ff, transpose(x_p(2:end)));

    % 5. Collect data for plotting
    dup_vec(j,:) =  my_ilc.kf_dpn.d;
    x_p_vec(j,:) =  x_p;
    x_b_vec(j,:) =  x_b;
    u_p_vec(j,:) =  u_p;
    u_des_vec(j,:) =  u_ff;
%     Simulation.plot_results(dt, F_vec_extra, x_b_extra, u_b_extra, x_p_extra, u_p_extra, dP_N_vec_extra, gN_vec_extra)
end

Simulation.plot_results(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
%% Plot trajectories over ILC iterations
close all
plotIterations(0.5*dt/m_p*dup_vec, "d_{P_N} through iterations", dt, 14)
plotIterations(disturbance, "real disturbance", dt)
% plotIterations(x_b_vec, "x_b through iterations", dt, 1)
% plotIterations(x_p_vec, "x_p through iterations", dt, 12)
% plotIterations(u_p_vec, "u_p through iterations", dt, 4)
% plotIterations(u_des_vec - u_des_vec(1,:), "u_{des} through iterations", dt, 4)

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
    timesteps = (1:n_x)*dt;
    plot(timesteps, y(1:every_n:end,:))
    legend(legend_vec)
    title(tititle)

end
