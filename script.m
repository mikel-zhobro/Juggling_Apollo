close all
dt = 0.002;
ta = 0; tb = 0.9032/2;
x_ta=-0.425; x_tb=0;
u_ta=0;     u_tb=4.4287; 
acc = [];
[x1, v1, a1, j1] = MinJerkTrajectory2.get_min_jerk_trajectory(dt, ta, tb, x_ta, x_tb, u_ta, u_tb);
% MinJerkTrajectory2.plot_paths(x1, v1, a1, j1, dt, 'eha')
%
tb2=tb+0.9032;
x_ta=0;     x_tb=0;
u_ta=4.4287;     u_tb=-u_ta/99; 
% acc.ta = a1(end);
[x2, v2, a2, j2] = MinJerkTrajectory2.get_min_jerk_trajectory(dt, tb, tb2, x_ta, x_tb, u_ta, u_tb, acc);

tb3=tb2+0.9032/2;
x_ta=0;         x_tb=-0.4;
u_ta=u_tb;   u_tb=0;
% acc.ta = a2(end);
% acc.tb = 0;
[x3, v3, a3, j3] = MinJerkTrajectory2.get_min_jerk_trajectory(dt, tb2, tb3, x_ta, x_tb, u_ta, u_tb, acc);

intervals = [ta/dt, tb/dt tb2/dt; tb/dt ,tb2/dt ,tb3/dt];
colors = {'red', [17 17 17]/255, [17 17 17]/255};
MinJerkTrajectory2.plot_paths([x1,x2,x3], [v1,v2,v3], [a1,a2,a3], [j1,j2,j3], dt,"Throw-free-catch", intervals, colors)
% 
% acc.ta=-9;
% [x, v, a, j] = MinJerkTrajectory2.get_min_jerk_trajectory(dt, ta, tb, x_ta, x_tb, u_ta, u_tb, acc);
% % MinJerkTrajectory2.plot_paths(x, v, a, j,dt,"checkup2")
% 
% 
% acc.ta=-9;
% acc.tb=12
% [x, v, a, j] = MinJerkTrajectory2.get_min_jerk_trajectory(dt, ta, tb, x_ta, x_tb, u_ta, u_tb, acc);
% % MinJerkTrajectory2.plot_paths(x, v, a, j,dt,"checkup3")

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

% ILC
kf_d1d2_params.M_diag = 0.1;                    % diagonal value of covarianc of noise on the measurment
kf_d1d2_params.P0_diag = 0.2;                   % the diagonal value of initial variance of disturbance
kf_d1d2_params.epsilon0 = 0.3;                  % initial variance of noise on the disturbance
kf_d1d2_params.epsilon_decrease_rate = 0.9;     % the decreasing factor of noise on the disturbance

kf_dpn_params.M_diag = 0.1;
kf_dpn_params.P0_diag = 0.1;
kf_dpn_params.epsilon0 = 0.1;
kf_dpn_params.epsilon_decrease_rate = 0.9;

my_ilc = ILC('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'dt', dt,              ...
             'x_ruhe', x_ruhe, 't_f', Tb, 'ub_0', ub_00,                         ...
             'kf_d1d2_params', kf_d1d2_params, 'kf_dpn_params', kf_dpn_params)  ;



% Iteration params
sim = Simulation('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'input_is_force', input_is_force, 'sys', my_ilc.sys, 'air_drag', true);
% ----------------------------------------------------------------------- %

%% Learn Throw
ILC_it = 125; % number of ILC iteration

% reset ilc
ub_0 = ub_00;
my_ilc.resetILC()

% Extra simulation to measure time of flight
T_sim_extra = 2*Tb;
N_sim_extra = Simulation.steps_from_time(T_sim_extra, dt);
    
% collect: dup, x_p, x_b, u_p
dup_vec = zeros([ILC_it, size(my_ilc.kf_dpn.d)]);
x_p_vec = zeros(ILC_it, my_ilc.N_1 + 1);
x_b_vec = zeros(ILC_it, N_sim_extra);
u_p_vec = zeros(ILC_it, my_ilc.N_1 + 1);
u_des_vec = zeros(ILC_it, my_ilc.N_1);
u_b0_vec = zeros(ILC_it, 1);
u_d2_vec = zeros(ILC_it, 1);
u_Tb_vec = zeros(ILC_it, 1);
% ILC Loop
close all
x0 = {x_ruhe; x_ruhe; 0; 0}; % the plate and ball in ruhe
[y_des,u_ff] = my_ilc.learnThrowStep(ub_0);
for j = 1:ILC_it
    display("ITERATION: " + num2str(j))
    
    % Main Simulation
    [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = sim.simulate_one_iteration_ss(dt, my_ilc.t_h/2, x0{:}, u_ff, 1);
    
    % Measurments for height and fly-time-to-zero
    % Extra simulation to measure time of flight
    [x_b_extra, u_b_extra, x_p_extra, u_p_extra, dP_N_vec_extra, gN_vec_extra, F_vec_extra] = sim.simulate_one_iteration(dt, 2*Tb, x_b(end), 0, u_b(end), 0, zeros(N_sim_extra,1), 1);
    
    hb_meas = max(x_b);
    fly_time_meas = find(x_b_extra(2:end)<=1e-5, 1 , 'first')*dt
    
    % LEARN THROW
    %                                                   from 2 since we cant optimize the first state x(0)
    [y_des,u_ff, ub_0] = my_ilc.learnThrowStep(ub_0, u_ff, transpose(x_p(2:end)), hb_meas, fly_time_meas);

    % 5. Collect data for plotting
    dup_vec(j,:) =  my_ilc.kf_dpn.d;
    x_p_vec(j,:) =  x_p;
    x_b_vec(j,:) =  x_b_extra;
    u_p_vec(j,:) =  u_p;
    u_des_vec(j,:) =  u_ff;
    u_d2_vec(j) =  my_ilc.kf_d1d2.d(2); %ub_0;
    u_b0_vec(j) =  ub_0;
    u_Tb_vec(j) = fly_time_meas;
%     Simulation.plot_results(dt, F_vec_extra, x_b_extra, u_b_extra, x_p_extra, u_p_extra, dP_N_vec_extra, gN_vec_extra)
end
iter_steps = 1:ILC_it;
tt = round(ILC_it/2);
figure; plot(iter_steps,[u_Tb_vec, Tb*ones(ILC_it,1)]); legend("Tb through iterations", "Tb")
figure; plot(iter_steps(tt:end),[u_b0_vec(tt:end)-u_b0_vec(end),u_d2_vec(tt:end), u_Tb_vec(tt:end)-Tb]); legend("ub0-ub0_{end} through iterations", "d2", "Tb_{meas}-Tb")
% Simulation.plot_results(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
% Simulation.plot_results(dt, F_vec_extra, x_b_extra, u_b_extra, x_p_extra, u_p_extra, dP_N_vec_extra, gN_vec_extra)
%% Plot trajectories over ILC iterations
close all
% plotIterations(dup_vec, "d_{P_N} through iterations", dt, 14)
plotIterations(x_b_vec, "x_b through iterations", dt, 1)
% plotIterations(x_p_vec, "x_p through iterations", dt, 1)
% plotIterations(u_p_vec, "u_p through iterations", dt, 4)
% plotIterations(u_des_vec - u_des_vec(1,:), "u_{des} through iterations", dt, 4)

%% TODO: Free move plate to catch
% Init State
% ub_throw = ub_0;
% x_b0 = 0;
% x_p0 = 0;       up_0 = ub_throw;      % acc.ta = -3*g;
% x_pTb = 0;      ub_T = -ub_0/6;       % acc.tb = 0;
%               %----important----%
% 
% x0 = [x_b0; x_p0; ub_0; up_0];

% Initialize lifted state space
% N_1= Simulation.steps_from_time(my_ilc.t_f, dt) - 1;
% impact_timesteps = ones(1, N_1);
% impact_timesteps([1,end]) = 2;
% ilc.resetILC(impact_timesteps)

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
