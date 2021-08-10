
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
[x_b, u_b, x_p, u_p, dP_N_vec, gN_vec] = simulate_one_iteration(dt, N, x_b0, x_p0, u_b0, u_p0, F_p);

%% Plotting for Simulation Example
intervals = find_continuous_intervals(find(gN_vec<0));

close all
figure
subplot(5,1,1)
plot(timesteps, x_b, 'r', timesteps, x_p, 'b')
plot_intervals(intervals, dt)
legend("Ball position", "Plate position")

subplot(5,1,2)
plot(timesteps, u_b, 'r', timesteps, u_p, 'b')
plot_intervals(intervals, dt)
legend("Ball velocity", "Plate velocity")

subplot(5,1,3)
plot(timesteps, dP_N_vec)
plot_intervals(intervals, dt)
legend("dP_N")

subplot(5,1,4)
plot(timesteps, gN_vec)
plot_intervals(intervals, dt)
legend("g_{N_{vec}}")

subplot(5,1,5)
plot(timesteps, F_p)
plot_intervals(intervals, dt)
legend("F_p")

%% Desired Trajectory planning example
% Design params
h_b_max = 1;                  % [m] maximal height the ball achievs

% Initialize disturbances
d1 = 0;
d2 = 0;
dup = 0;

% Initialize throw point
x_b0 = 0;
x_p0 = x_b0;
x_pTb = x_p0;
[xp_des, T] = compute_desired_trajectory(h_b_max, d1, d2, x_b0, x_pTb);

close all
figure
timesteps = 0:dt:T;
subplot(4,1,1)
plot(timesteps, xp_des(1,:))
legend("Plate position")

subplot(4,1,2)
plot(timesteps, xp_des(2,:))
legend("Plate velocity")

subplot(4,1,3)
plot(timesteps, xp_des(3,:))
legend("Plate acceleration")

subplot(4,1,4)
plot(timesteps, xp_des(4,:))
legend("Plate jerk")


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
    % 1. Plan desired Plate trajectory
    [xp_des] = compute_desired_trajectory(h_b_max, d1, d2, x_b0, x_pTb);

    % 2. Compute optimal input velocities u_des
    [u_des] = compute_optimal_input_signal(xp_des, dup);

    % 3. Simulate the calculated inputs
    [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec] = simulate_one_iteration(dt, N, x_b0, x_p0, u_b0, u_b0, u, false);

    % 4.Identify errors d1, d2, dp_j (kalman filter or optimization problem
    [d1, d2, dup] = filter_disturbances(x_b, u_b, x_p, u_p, d1, d2, dup);
end


%% Components
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

%% Helpers
% Find intervals where gN<=0
function intervals = find_continuous_intervals(indices)
    last = indices(1);
    start = last;
    starts = [];
    ends = [];
    for i=indices(1:end)
        if i-last>1
            starts(end+1) = start;
            ends(end+1) = last;
            start = i;
        end
        last = i;
    end
    intervals = [starts; ends];
end

function plot_intervals(intervals, dt)
    for i = intervals
        patch(dt*[i(1) i(1), i(2) i(2)], [min(ylim) max(ylim) max(ylim) min(ylim)], [91, 207, 244]/255, 'LineStyle', 'none', 'FaceAlpha', 0.3 )
    end
end
