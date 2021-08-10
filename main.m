

%% Parameters
% Constants
global g m_b m_p k_c;
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

%% Simulation
% Input
A = 0.3;                                    % [m] amplitude
timesteps = dt * (0:N);                     % [s,s,..] timesteps
F_p = 100 * m_p * A*sin(2*pi/T *timesteps); % [N] input force on the plate

% Simulation for N steps
[x_b, u_b, x_p, u_p, dP_N_vec, gN_vec] = simulate_one_iteration(dt, N, x_b0, x_p0, u_b0, u_p0, F_p);

%% Plotting
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

%% Feedforward controled system
% Design params
dt = 0.01;                    % [s] discretization timestep
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

% 1. Dynamics
function F_p = force_from_velocity(u_des_p, u_p)
    global m_p k_c;
    F_p  = m_p * k_c * (u_des_p-u_p);
end

function [x_b_new, x_p_new, u_b_new, u_p_new, dP_N, gN] = simulate_one_step(dt, F_p_i, x_b_i, x_p_i, u_b_i, u_p_i)
    global g m_b m_p;

    x_b_1_2 = x_b_i + 0.5*dt*u_b_i;
    x_p_1_2 = x_p_i + 0.5*dt*u_p_i;

    gN = x_b_1_2 - x_p_1_2;
    gamma_n_i = u_b_i - u_p_i;
    if gN <=0
        dP_N = max(0,(-gamma_n_i + g*dt + F_p_i*dt/m_p)/ (m_b^-1 + m_p^-1));
    else
        dP_N = 0;
    end

    u_b_new = u_b_i - g*dt + dP_N/m_b;
    u_p_new = u_p_i + F_p_i*dt/m_p - dP_N/m_p;

    x_b_new = x_b_1_2 + u_b_new*dt/2;
    x_p_new = x_p_1_2 + u_p_new*dt/2;
end

function [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec] = simulate_one_iteration(dt, N, x_b0, x_p0, u_b0, u_p0, u, input_is_force)
    % if otherwise not specified input is force
    if nargin<8
      input_is_force=true;
    end

    % Initialize state vectors of the system
    x_b = zeros(1,N+1); x_b(1) = x_b0;
    u_b = zeros(1,N+1); u_b(1) = u_b0;
    x_p = zeros(1,N+1); x_p(1) = x_p0;
    u_p = zeros(1,N+1); u_p(1) = u_p0;

    % Initialize helpers used to gather info for debugging
    dP_N_vec = zeros(1,N+1);
    gN_vec = zeros(1,N+1);

    % Simulation
    for i = (1:N)
        % one step simulation
        if input_is_force
            F_p = u(i);
        else
            F_p = force_from_velocity(u(i), u_p(i));
        end
        [x_b_new, x_p_new, u_b_new, u_p_new, dP_N, gN] = simulate_one_step(dt, F_p, x_b(i), x_p(i), u_b(i), u_p(i));

        % collect state of the system
        x_b(i+1) = x_b_new;
        x_p(i+1) = x_p_new;
        u_b(i+1) = u_b_new;
        u_p(i+1) = u_p_new;

        % collect helpers
        dP_N_vec(i) = dP_N;
        gN_vec(i) = gN;
    end
end

% 2. Plan desired Trajectory
function [Tb, ub_0] = plan_ball_trajectory(hb, d1, d2)
    global g;
    ub_0 = sqrt(2*g*(hb - d1));  % velocity of ball at throw point
    Tb = 2*ub_0/g + d2; % flying time of the ball
end

function [xp_des] = plan_plate_trajectory(Tb, ub_0, ub_Tb, x_b0, x_pTb)
    % xb_0, ub_0: conditions at t=0
    % xb_Tb, ub_T: conditions at t=Tb  [assumed 0]
    % xp_des(t) = [x_p(t); u_p(t)]
    
    % calculate the coeficients
    C_cv = [10/(7*Tb^2), -120/(7*Tb^4), 240/(7*Tb^5);
            3/(7*Tb), -120/(7*Tb^3), 240/(7*Tb^4);
               Tb/84,     -8/(7*Tb),  30/(7*Tb^2)];
    v = [0, ub_Tb-ub_0; -ub_0*Tb];
    c = C_cv * v;
    c1 = c(1);
    c2 = c(2);
    c4 = c(3);
    c5 = ub_0;
    
    % calculate desired plate input
    t = 1:Tb;
    u = -c1*t.^2/2 + c2*t;                          % jerk of the plate for t=0 -> t=Tb
    a_p = c1*t^3/6 - c2*t^2/2 + c4;                 % acceleration of the plate for t=0 -> t=Tb
    u_p = c1*t^4/24 - c2*t^3/6 + c4*t;              % velocity of the plate for t=0 -> t=Tb
    x_p = c1*t^5/120 - c2*t^4/24 + c4*t^2/2 + c5*t; % position of the plate for t=0 -> t=Tb
    % for t=Tb -> t=2Tb we just take the symmetric negative of the
    % corresponding trajectory in t=0 -> t=Tb
    xp_des = [x_p, x_p(end:-1:1)*-1; u_p, u_p(end:-1:1)*-1];
end

function [xp_des] = compute_desired_trajectory(hb, d1, d2, x_b0, x_pTb)
    % 0. Plan Ball Trajectory
    [Tb, ub_0] = plan_ball_trajectory(hb, d1, d2);

    % 1. Calculate Desired Platte Trajectory (min jerk trajectory)
    [xp_des] = plan_plate_trajectory(Tb, ub_0, ub_0, x_b0, x_pTb);
end

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
