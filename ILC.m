classdef ILC < matlab.System
    % ILC('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'dt', dt, 'x_ruhe', x_ruhe, 't_f', t_f)

    % objects with methods to be called
    properties
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
        x_ruhe % starting position of the plate
        % t_f is calculated depending on the below params
        %     based on the formula n_b/n_a = (t_h+t_f)/(t_h+t_e)
        t_f % flying time of the ball
        t_h % time that ball is in the hand
        t_e % TODO: time that hand is empty
        n_h % number of hands (1 or 2)
        n_b % number of balls (1,2,3..)
        N_1
    end
    
    %state
    properties
        ub_0
    end

    methods
        function ilc = ILC(varargin)
            setProperties(ilc,nargin,varargin{:})
            ilc.initTimes()
            ilc.initILC()
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
            ilc.lss = LiftedStateSpace('Ad', Ad, 'Bd', Bd,        ...
                                       'Cd', Cd, 'S', S,          ...
                                       'x0', [ilc.x_ruhe; ilc.x_ruhe; 0; 0], ...
                                       'c', c,  ...
                                       'Ad_impact', Ad_impact,    ...
                                       'Bd_impact', Bd_impact,    ...
                                       'c_impact', c_impact       );

            % V. DESIRED INPUT OPTIMIZER
            ilc.quad_input_optim = OptimizationDesiredInput(ilc.lss);

            % VI. KALMAN FILTERS
            % d1, d2
            d1d2 = [0;0];
            n_d1d2 = 2;
            s.GK = eye(n_d1d2); s.Gd0 = [0; 0]; s.GF = [0;0];
            ilc.kf_d1d2 = ILCKalmanFilter('lss', s, 'dt', ilc.dt, 'd0', d1d2,   ...
                                          'M' , 0.1 * eye(n_d1d2, n_d1d2),  ...
                                          'P0', 1e-2*eye(n_d1d2, n_d1d2),   ...
                                          'epsilon0', 0.3                   );
            % dpn
            ilc.N_1= Simulation.steps_from_time(ilc.t_h/2, ilc.dt) - 1;  % important since input cannot influence the first state
            dup = zeros(n_dup*ilc.N_1,1);
            ilc.kf_dpn = ILCKalmanFilter('lss', ilc.lss,                 ...
                                         'M' , 0.1 * eye(n_y*ilc.N_1, n_y*ilc.N_1),         ...
                                         'P0', 1e-2*eye(n_dup*ilc.N_1, n_dup*ilc.N_1),      ...
                                         'dt', ilc.dt, 'd0', dup, 'epsilon0', 0.3       );
        end
        
        function resetILC(ilc, impact_timesteps)
            % reset LSS
            if nargin>1
                ilc.lifted_state_space.updateQuadrProgMatrixes(impact_timesteps)
            end

            % reset KFs
            ilc.kf_d1d2.resetKF()
            ilc.kf_dpn.resetKF()
        end
        
        function u_des = getDesiredInput(ilc, y_des)
            u_des = ilc.quad_input_optim.calcDesiredInput(ilc.kf_dpn.d, y_des);
        end
        
        function [y_des,u_des] = learnThrowStep(ilc, u_des, y_meas, hb_meas, fly_time_meas)
            if nargin > 1 % the first time we call it like: learnThrowStep(ilc)
                % calc d1d2 disturbances
                d1_meas = hb_meas - 0.5*ilc.ub_0^2/ilc.g; % = hb_meas - ub0^2/(2g)
                d2_meas = fly_time_meas - ilc.t_f; % = Tb_meas - 2ub0/g
                ilc.kf_d1d2.updateStep(0, [d1_meas; d2_meas]);

                % calc dpn disturbance
                ilc.kf_dpn.updateStep(u_des, y_meas);
            else
                ilc.lss.updateQuadrProgMatrixes(2*ones(1, ilc.N_1))
            end
            
            % calc new ub_0
%             ilc.ub_0 = 0.5*ilc.g*( ilc.t_f - ilc.kf_d1d2.d(2));
            ilc.ub_0 = ilc.ub_0 - 0.7*ilc.kf_d1d2.d(2); % move in oposite direction of error

            % new MinJerk
            [y_des, v_des, a_des, j_des] = MinJerkTrajectory2.get_min_jerk_trajectory(ilc.dt, 0, ilc.t_h/2, ilc.x_ruhe, 0, 0, ilc.ub_0);
%             MinJerkTrajectory2.plot_paths(y_des, v_des, a_des, j_des, ilc.dt, 'MinJerk Free start-end acceleration')
            
            % calc desired input
            u_des = ilc.getDesiredInput(transpose(y_des(2:end)));
        end
        
        function [Tb, ub_0] = plan_ball_trajectory(ilc, hb, Fp)
            ub_0 = sqrt(2*ilc.g*(hb - ilc.kf_d1d2.d(1)));  % velocity of ball at throw point
            Tb = 2*ub_0/ilc.g + ilc.kf_d1d2.d(2); % flying time of the ball
            if nargin > 3 % consider impuls
                dPN = ilc.m_b*ilc.m_p/(ilc.m_b+ilc.m_p) * (ilc.g*ilc.dt + Fp*ilc.dt/ilc.m_p);
                ub_0 = ub_0 - dPN;
            end
        end
        
    end
end

