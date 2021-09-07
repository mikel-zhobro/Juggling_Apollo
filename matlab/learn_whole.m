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
Th = Tb/2;
TT = Tb+Th;
% Init state
x_ruhe = -0.4;
x0 = {x_ruhe; x_ruhe; 0; 0}; % the plate and ball in ruhe

% ILC
kf_d1d2_params.M_diag = 0.1;                    % diagonal value of covarianc of noise on the measurment
kf_d1d2_params.P0_diag = 0.2;                   % the diagonal value of initial variance of disturbance
kf_d1d2_params.epsilon0 = 0.3;                  % initial variance of noise on the disturbance
kf_d1d2_params.epsilon_decrease_rate = 0.9;     % the decreasing factor of noise on the disturbance

kf_dpn_params.M_diag = 0.1;
kf_dpn_params.P0_diag = 0.1;
kf_dpn_params.epsilon0 = 0.1;
kf_dpn_params.epsilon_decrease_rate = 1;

my_ilc = ILC('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'dt', dt,              ...
             'x_0', cell2mat(x0), 't_f', Tb, 'input_is_force', input_is_force,  ...
             'kf_d1d2_params', kf_d1d2_params, 'kf_dpn_params', kf_dpn_params)  ;

% my_ilc.resetILC()

% Iteration params
sim = Simulation('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'input_is_force', input_is_force, 'sys', my_ilc.sys, 'air_drag', true);
% ----------------------------------------------------------------------- %

%% Learn Throw
close all
ILC_it = 122; % number of ILC iteration
ub_0 = ub_00;
% reset ilc
[y_des,u_ff] = my_ilc.learnWhole(ub_00);
my_ilc.resetILC()
plot(u_ff)
% figure
% plot(y_des)
% collect: dup, x_p, x_b, u_p
dup_vec = zeros([ILC_it, size(my_ilc.kf_dpn.d)]);
x_p_vec = zeros(ILC_it, my_ilc.N_1 + 1);
u_p_vec = zeros(ILC_it, my_ilc.N_1 + 1);
u_des_vec = zeros(ILC_it, my_ilc.N_1);
u_b0_vec = zeros(ILC_it, 1);
u_d2_vec = zeros(ILC_it, 1);
u_Tb_vec = zeros(ILC_it, 1);

% ILC Loop
% close all
% disturbance to be learned
N_temp = round(my_ilc.N_1/3);
period = 0.02/dt;
disturbance = 200*sin(2*pi/period*(0:my_ilc.N_1-1)); % disturbance on the plate position
for j = 1:ILC_it
    display("ITERATION: " + num2str(j))

    % Main Simulation
    [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = sim.simulate_one_iteration(dt, my_ilc.t_h+my_ilc.t_f, x0{:}, u_ff, 1, disturbance);

    hb_meas = max(gN_vec);
    fly_time_meas = ( N_temp + find(gN_vec(N_temp:end)<=1e-5, 1 , 'first'))*dt
%     fly_time_meas = find(x_b_extra(2:end)<=1e-5, 1 , 'first')*dt

    % LEARN THROW
    %                                                   from 2 since we cant optimize the first state x(0)
    [y_des,u_ff, ub_0] = my_ilc.learnWhole(ub_0, u_ff, transpose(x_p(2:end)), hb_meas, fly_time_meas);

    % 5. Collect data for plotting
    dup_vec(j,:) =  my_ilc.kf_dpn.d;
    x_p_vec(j,:) =  x_p;
    u_p_vec(j,:) =  u_p;
    u_des_vec(j,:) =  u_ff;
    u_d2_vec(j) =  my_ilc.kf_d1d2.d(2); %ub_0;
    u_b0_vec(j) =  ub_0;
    u_Tb_vec(j) = fly_time_meas;
end
Simulation.plot_results(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)

%%
iter_steps = 1:ILC_it;
tt = round(ILC_it/2);
figure; plot(iter_steps,[u_Tb_vec, Tb*ones(ILC_it,1)]); legend("Tb through iterations", "Tb")
% figure; plot(iter_steps,[u_Tb_vec - Tb*ones(ILC_it,1)]); legend("Tb through iterations", "Tb")
figure; plot(iter_steps(tt:end),[u_b0_vec(tt:end)-u_b0_vec(end),u_d2_vec(tt:end), u_Tb_vec(tt:end)-Tb]); legend("ub0-ub0_{end} through iterations", "d2", "Tb_{meas}-Tb")
% Simulation.plot_results(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
%% Plot trajectories over ILC iterations
close all
plotIterations([dup_vec; x_p(2:end)-y_des(2:end)] , "d_{p} through iterations", dt, 1)
% plotIterations(x_b_vec, "x_b through iterations", dt, 3)
plotIterations([x_p_vec; y_des], "x_p through iterations", dt, 1)
% plotIterations(u_p_vec, "u_p through iterations", dt, 2)

% plotIterations(u_des_vec, "u_{ff} through iterations", dt, 2)
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
%     legend_vec(end+1) = "last iteration";
    figure
    hold on
    timesteps = (1:n_x)*dt;
    plot(timesteps, y(1:every_n:end,:))
%     plot(timesteps, y(end,:), '*')
    legend(legend_vec)
    title(tititle)

end
