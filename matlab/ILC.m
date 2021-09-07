classdef ILC < matlab.System
    % ILC('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'dt', dt, 'x_0', x_0, 't_f', t_f)

    % objects with methods to be called
    properties
        input_is_force
        lss % lifted_state_space
        quad_input_optim
        kf_d1d2
        kf_dpn
    end

    % constant properties
    properties
        m_b
        m_p
        k_c
        g
        dt
        sys
    end

    % design params
    properties
        kf_d1d2_params
        kf_dpn_params
        x_0 % starting position of the plate
        % t_f is calculated depending on the below params
        %     based on the formula n_b/n_a = (t_h+t_f)/(t_h+t_e)
        t_f % flying time of the ball
        t_h % time that ball is in the hand
        t_e % TODO: time that hand is empty
        n_h % number of hands (1 or 2)
        n_b % number of balls (1,2,3..)
        N_1 % time steps
        y_des
    end

    methods
        function ilc = ILC(varargin)
            setProperties(ilc,nargin,varargin{:})
            ilc.initTimes()
        end

        function initTimes(ilc)
            % TODO
            % ilc.t_f = ilc.n_b/ilc.n_h *(ilc.t_h+ilc.t_e)-ilc.t_h;
            ilc.t_h = ilc.t_f/2;
        end

        function initILC(ilc)
            % Here we want to set some convention to avoid missunderstandins later on.
            % 1. the state is [xb, xp, ub, up]^T
            % 2. the system can have as input either velocity u_des or the force F_p

            % II. SYSTEM DYNAMICS
            input_is_force = false;
            ilc.sys = DynamicSystem('m_b', ilc.m_b, 'm_p', ilc.m_p, 'k_c', ilc.k_c, 'g', ilc.g, 'dt', ilc.dt);
            if input_is_force
                [Ad, Bd, Cd, S, c] = ilc.sys.getSystemMarixesForceControl(ilc.dt, false);
                [Ad_impact, Bd_impact, ~, ~, c_impact] = ilc.sys.getSystemMarixesForceControl(ilc.dt, true);

            else
                [Ad, Bd, Cd, S, c] = ilc.sys.getSystemMarixesVelocityControl(ilc.dt, false);
                [Ad_impact, Bd_impact, ~, ~, c_impact] = ilc.sys.getSystemMarixesVelocityControl(ilc.dt, true);
            end
            % Sizes
            n_x = size(Ad, 1);
            n_y = size(Cd, 1);
            n_dup = size(S, 2);

            % III. LIFTED STATE SPACE
            ilc.lss = LiftedStateSpace('Ad', Ad, 'Bd', Bd,      ...
                                       'Cd', Cd, 'S', S,        ...
                                       'x0', ilc.x_0,           ...
                                       'c', c,                  ...
                                       'Ad_impact', Ad_impact,  ...
                                       'Bd_impact', Bd_impact,  ...
                                       'c_impact', c_impact     );

            % V. DESIRED INPUT OPTIMIZER
            ilc.quad_input_optim = OptimizationDesiredInput(ilc.lss);

            % VI. KALMAN FILTERS
            % d1, d2
            d1d2 = [0;0];
            n_d1d2 = 2;
            s.GK = eye(n_d1d2); s.Gd0 = [0; 0]; s.GF = [0;0];
            ilc.kf_d1d2 = ILCKalmanFilter('lss', s, 'dt', ilc.dt, 'd0', d1d2,                                   ...
                                          'M' , ilc.kf_d1d2_params.M_diag * eye(n_d1d2, n_d1d2),                ...
                                          'P0', ilc.kf_d1d2_params.P0_diag*eye(n_d1d2, n_d1d2),                 ...
                                          'epsilon0', ilc.kf_d1d2_params.epsilon0,                              ...
                                          'epsilon_decrease_rate', ilc.kf_d1d2_params.epsilon_decrease_rate)    ;
            % dpn
            dup = zeros(n_dup*ilc.N_1,1);
            ilc.kf_dpn = ILCKalmanFilter('lss', ilc.lss, 'dt', ilc.dt, 'd0', dup,                           ...
                                         'M' , ilc.kf_dpn_params.M_diag * eye(n_y*ilc.N_1, n_y*ilc.N_1),    ...
                                         'P0', ilc.kf_dpn_params.P0_diag*eye(n_dup*ilc.N_1, n_dup*ilc.N_1), ...
                                         'epsilon0', ilc.kf_dpn_params.epsilon0,                            ...
                                         'epsilon_decrease_rate', ilc.kf_dpn_params.epsilon_decrease_rate)  ;
        end

        function resetILC(ilc, impact_timesteps)
            % reset LSS
            if nargin > 1
                ilc.lss.updateQuadrProgMatrixes(impact_timesteps)
            end

            % reset KFs
            ilc.kf_d1d2.resetKF()
            ilc.kf_dpn.resetKF()
        end

        %% 1. Throw
        function [y_des,u_ff_new, ub_0] = learnThrowStep(ilc, ub_0, u_ff_old, y_meas, hb_meas, fly_time_meas)
            if nargin > 2 % the first time we call it like: learnThrowStep(ilc)
                % estimate d1d2 disturbances
                d1_meas = hb_meas - 0.5*ub_0^2/ilc.g; % = hb_meas - ub0^2/(2g)
                d2_meas = fly_time_meas - ilc.t_f; % = Tb_meas - 2ub0/g
                ilc.kf_d1d2.updateStep(0, [d1_meas; d2_meas]);

                % estimate dpn disturbance
                ilc.kf_dpn.updateStep(u_ff_old, y_meas);
            else
                ilc.N_1 = Simulation.steps_from_time(ilc.t_h/2, ilc.dt) - 1;  % important since input cannot influence the first state
                ilc.initILC()
                ilc.resetILC(2*ones(1, ilc.N_1))
            end

            % calc new ub_0
%             ub_0 = 0.5*ilc.g*( ilc.t_f - ilc.kf_d1d2.d(2));
            ub_0 = ub_0 - 0.3*0.5*ilc.g*ilc.kf_d1d2.d(2); % move in oposite direction of error
%             ub_0 = ub_0 - 0.7*ilc.kf_d1d2.d(2); % move in oposite direction of error

            % new MinJerk
            [y_des, v_des, a_des, j_des] = MinJerkTrajectory2.get_min_jerk_trajectory(ilc.dt, 0, ilc.t_h/2, ilc.x_0(1), 0, 0, ub_0);
%             MinJerkTrajectory2.plot_paths(y_des, v_des, a_des, j_des, ilc.dt, 'MinJerk Free start-end acceleration')

            % calc desired input
            u_ff_new = ilc.quad_input_optim.calcDesiredInput(ilc.kf_dpn.d, transpose(y_des(2:end)));
        end

        %% 2. Plate Free Motion
        function [u_ff_new] = learnPlateMotionStep(ilc, y_des, u_ff_old, y_meas)
            if nargin >2 % the first time we call it like: learnPlateMotionStep(ilc)
                % estimate dpn disturbance
                ilc.kf_dpn.updateStep(u_ff_old, y_meas);
            else
                ilc.N_1 = Simulation.steps_from_time(ilc.t_f, ilc.dt) - 1;  % important since input cannot influence the first state
                ilc.initILC()
                impact_timesteps = ones(1, ilc.N_1);
                impact_timesteps([1,end]) = 2;
                ilc.resetILC(impact_timesteps);
            end

            % calc desired input
            u_ff_new = ilc.quad_input_optim.calcDesiredInput(ilc.kf_dpn.d, transpose(y_des));
        end

        %% 3. Whole
        function [y_des,u_ff_new, ub_0] = learnWhole(ilc, ub_0, u_ff_old, y_meas, hb_meas, fly_time_meas)
            if nargin > 2 % the first time we call it like: learnThrowStep(ilc)
                % estimate d1d2 disturbances
                d1_meas = hb_meas - 0.5*ub_0^2/ilc.g; % = hb_meas - ub0^2/(2g)
                d2_meas = fly_time_meas - ilc.t_f; % = Tb_meas - 2ub0/g
                ilc.kf_d1d2.updateStep(0, [d1_meas; d2_meas]);

                % estimate dpn disturbance
                ilc.kf_dpn.updateStep(u_ff_old, y_meas);
            else
                ilc.N_1 = Simulation.steps_from_time(ilc.t_h + ilc.t_f, ilc.dt) - 1;  % important since input cannot influence the first state
                ilc.initILC()
                ilc.resetILC(2*ones(1, ilc.N_1))
%                 timesteps_impact = 2*ones(1, ilc.N_1);
%                 timesteps_impact(round(ilc.N_1/4):round(3*ilc.N_1/4)) = 1;
%                 ilc.resetILC(timesteps_impact)
            end

            % calc new ub_0
%             ub_0 = 0.5*ilc.g*( ilc.t_f - ilc.kf_d1d2.d(2));
            ub_0 = ub_0 - 0.1*0.5*ilc.g*ilc.kf_d1d2.d(2); % move in oposite direction of error
%             ub_0 = ub_0 - 0.7*ilc.kf_d1d2.d(2); % move in oposite direction of error

            % new MinJerk
            % new MinJerk
            t0 = 0;            t1 = ilc.t_h/2;   t2 = t1 + ilc.t_f;    t3 = ilc.t_f + ilc.t_h;
            x0 = ilc.x_0(1);   x1 = 0;           x2 = 0;               x3 = x0;
            u0 = ilc.x_0(3);   u1 = ub_0;        u2 = -ub_0/6;         u3 = u0;
            [y1, v1, a1, j1] = MinJerkTrajectory2.get_min_jerk_trajectory(ilc.dt, t0, t1, x0, x1, u0, u1);
            t1 = (length(y1))*ilc.dt;
            x1 = y1(end);
            u1 = v1(end);
            [y2, v2, a2, j2] = MinJerkTrajectory2.get_min_jerk_trajectory(ilc.dt, t1, t2, x1, x2, u1, u2);
            t2 = t1 +(length(y2))*ilc.dt;
            x2 = y2(end);
            u2 = v2(end);
            [y3, v3, a3, j3] = MinJerkTrajectory2.get_min_jerk_trajectory(ilc.dt, t2, t3, x2, x3, u2, u3);
%
            ilc.y_des = [y1, y2, y3];
            y_des = ilc.y_des;
%             v_des = [v1, v2, v3];
%             a_des = [a1, a2, a3];
%             j_des = [j1, j2, j3];
%             MinJerkTrajectory2.plot_paths(y_des, v_des, a_des, j_des, ilc.dt, 'MinJerk Free start-end acceleration')
            % calc desired input
            u_ff_new = ilc.quad_input_optim.calcDesiredInput(ilc.kf_dpn.d, transpose(y_des(2:end)));
        end

    end
end

