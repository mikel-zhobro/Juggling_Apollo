classdef Simulation < matlab.System
    %SIMULATION Summary of this class goes here
    %   Detailed explanation goes here
    properties
        m_b;                % mass of ball
        m_p;                % mass of plate
        k_c;                % force coefficient
        g;                  % gravitational acceleration constant
        input_is_force;     % true if input is force, false if input is velocity
        sys;                % dynamic sys used to get state space matrixes of system
        air_drag;           % bool whether we should add air drag to the ball
        plate_cos_dis       % bool whether we should add some cosinus form disturbance on the plate trajectory
    end

   methods
    % Constructor
    function obj = Simulation(varargin)
        setProperties(obj,nargin,varargin{:})
    end

    function F_p = force_from_velocity(obj, u_des_p, u_p)
        F_p  = obj.m_p * obj.k_c * (u_des_p-u_p);
    end

    function [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, u_vec] = simulate_one_iteration(obj, dt, T, x_b0, x_p0, u_b0, u_p0, u, repetitions, d)
        if nargin < 9
            repetitions = 1;
        end
        if nargin < 10
            d = zeros(size(u)); % disturbance
        end
        u = repmat(u,repetitions,1);
        d = repmat(d,repetitions,1);
        % Vectors to collect the history of the system states
        N = Simulation.steps_from_time(T, dt)*repetitions;
        x_b = zeros(1,N); x_b(1) = x_b0;
        u_b = zeros(1,N); u_b(1) = u_b0;
        x_p = zeros(1,N); x_p(1) = x_p0;
        u_p = zeros(1,N); u_p(1) = u_p0;
        % Vector to collect extra info for debugging
        dP_N_vec = zeros(1,N);
        gN_vec = zeros(1,N);
        u_vec = zeros(1,N);

        % Simulation
        for i = (1:N-1)
            % one step simulation
            [x_b_new, x_p_new, u_b_new, u_p_new, dP_N, gN, u_i] = obj.simulate_one_step(dt, u(i), x_b(i), x_p(i), u_b(i), u_p(i), d(i));
            % collect state of the system
            x_b(i+1) = x_b_new;
            x_p(i+1) = x_p_new;
            u_b(i+1) = u_b_new;
            u_p(i+1) = u_p_new;
            % collect helpers
            dP_N_vec(i) = dP_N;
            gN_vec(i) = gN;
            u_vec(i) = u_i;
        end
    end

    function [x_b_new, x_p_new, u_b_new, u_p_new, dP_N, gN, F_i] = simulate_one_step(obj, dt, u_i, x_b_i, x_p_i, u_b_i, u_p_i, di)
        % this works only with force as input
        % so if we get speed we transform it to force.
        F_i = u_i;
        if ~obj.input_is_force % did we get speed?
            F_i = obj.force_from_velocity(u_i, u_p_i);
        end

        x_b_1_2 = x_b_i + 0.5*dt*u_b_i;
        x_p_1_2 = x_p_i + 0.5*dt*u_p_i;

        % gN = x_b_1_2 - x_p_1_2;
        gN = x_b_i - x_p_i;
        gamma_n_i = u_b_i - u_p_i;
        contact_impact = gN <=1e-5 && (((-gamma_n_i + obj.g*dt + u_i*dt/obj.m_p))>=0);
        if contact_impact
            dP_N = max(0,(-gamma_n_i + obj.g*dt + F_i*dt/obj.m_p)/ (obj.m_b^-1 + obj.m_p^-1));
            % dP_N = (-gamma_n_i + obj.g*dt + u_i*dt/obj.m_p)/ (obj.m_b^-1 + obj.m_p^-1);
        else
            dP_N = 0;
        end

        state_friction = obj.get_state_friction([x_b_i; x_p_i; u_b_i; u_p_i], dt);
        state_disturbance = Simulation.get_state_disturbance(contact_impact, di);
        
        u_b_new = u_b_i - obj.g*dt + dP_N/obj.m_b           + state_friction(3) + state_disturbance(3);
        u_p_new = u_p_i + F_i*dt/obj.m_p - dP_N/obj.m_p     + state_friction(4) + state_disturbance(4);
        x_b_new = x_b_1_2 + 0.5*dt*u_b_new                  + state_friction(1) + state_disturbance(1);
        x_p_new = x_p_1_2 + 0.5*dt*u_p_new                  + state_friction(2) + state_disturbance(2);
    end

    function [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, u_vec] = simulate_one_iteration_ss(obj, dt, T, x_b0, x_p0, u_b0, u_p0, u, repetitions, d)
        if nargin < 9
            repetitions = 1;
        end
        if nargin < 10
            d = zeros(size(u)); % disturbance
        end
        u = repmat(u,repetitions,1);
        d = repmat(d,repetitions,1);
        % Vectors to collect the history of the system states
        N = Simulation.steps_from_time(T, dt)*repetitions;
        x_b = zeros(1,N); x_b(1) = x_b0;
        u_b = zeros(1,N); u_b(1) = u_b0;
        x_p = zeros(1,N); x_p(1) = x_p0;
        u_p = zeros(1,N); u_p(1) = u_p0;
        % Vector to collect extra info for debugging
        dP_N_vec = zeros(1,N);
        gN_vec = zeros(1,N);
        u_vec = zeros(1,N);

        % Simulation
        for i = (1:N-1)
            % Simlate using matrixes
            [x_b_new, x_p_new, u_b_new, u_p_new, dP_N, gN, u_i] = obj.state_space_one_step(dt, u(i), x_b(i), x_p(i), u_b(i), u_p(i), d(i));
            % collect states
            x_b(i+1) = x_b_new;
            x_p(i+1) = x_p_new;
            u_b(i+1) = u_b_new;
            u_p(i+1) = u_p_new;
            % collect helpers
            dP_N_vec(i) = dP_N;
            gN_vec(i) = gN;
            u_vec(i) = u_i;
        end
    end

    function [x_b_new, x_p_new, u_b_new, u_p_new, dP_N, gN, u_ii] = state_space_one_step(obj, dt, u_i, x_b_i, x_p_i, u_b_i, u_p_i, di)
        % gN = x_b_i + 0.5*dt*u_b_i - (x_p_i + 0.5*dt*u_p_i);
        gN = x_b_i - x_p_i;
        gamma_n_i = u_b_i-u_p_i;

        if obj.input_is_force
            contact_impact = (gN<=1e-5) && (((-gamma_n_i + obj.g*dt + u_i*dt/obj.m_p))>=0);
            [Ad, Bd, ~, ~, c] = obj.sys.getSystemMarixesForceControl(dt, contact_impact);
        else
            contact_impact = (gN<=1e-5) && (((-gamma_n_i + obj.g*dt + obj.force_from_velocity(u_i, u_p_i)*dt/obj.m_p))>0);
            [Ad, Bd, ~, ~, c] = obj.sys.getSystemMarixesVelocityControl(dt, contact_impact);
        end

        state_disturbance = Simulation.get_state_disturbance(contact_impact, di);
        state_friction = obj.get_state_friction([x_b_i; x_p_i; u_b_i; u_p_i], dt);

        state_new = Ad*[x_b_i; x_p_i; u_b_i; u_p_i] + Bd*u_i + c + state_friction + state_disturbance;

        x_b_new = state_new(1);
        x_p_new = state_new(2);
        u_b_new = state_new(3);
        u_p_new = state_new(4);

        dP_N = 0;
        u_ii = u_i;
        if ~obj.input_is_force
            u_ii = obj.force_from_velocity(u_i, u_p_i);
        end
        if contact_impact
            dP_N = max(0,(-gamma_n_i + obj.g*dt + u_ii*dt/obj.m_p)/ (obj.m_b^-1 + obj.m_p^-1));
        end
    end

    function state_friction=get_state_friction(obj, x0, dt)
        % F_D is force caused by air drag
        state_friction = zeros(size(x0));
        if obj.air_drag
            F_D = Simulation.friction(x0(3));
            du = -F_D*dt/obj.m_b;
            state_friction(1) = 0.5*du*dt;
            state_friction(3) = du;
        end
    end
    

   end

   %% Static Helpers
   methods (Static)
    function state_disturbance=get_state_disturbance(contact_impact, di)
        % F_D is force caused by air drag
        state_disturbance = [0;di;0;0];
        if contact_impact
            state_disturbance(1) = di;
        end
    end
    
    function f_drag=friction(v)
        % D is the diameter of the ball
        % c = 1/4*p*A = pi/16*p*D^2
        D = 0.4; % ball has diameter of 5cm
        p = 1.225; % [kg/m]  air density
        c = pi/16*p*D^2;
        f_drag = sign(v)*c*v^2;
    end

    function N = steps_from_time(T, dt)
        N = floor(T/dt)+1;
    end

    function intervals = find_continuous_intervals(indices)
        intervals = [];
        if ~isempty(indices)
            % Find intervals where gN<=0
            last = indices(1);
            start = last;
            starts = [];
            ends = [];
            for i=indices(2:end)
                if i-last>1
                    starts(end+1) = start-1;
                    ends(end+1) = last;
                    start = i;
                end
                last = i;
                if i==indices(end)
                    starts(end+1) = start-1;
                    ends(end+1) = last;
                    start = i;
                end
            end
            intervals = [starts; ends];
        end
    end

    function plot_results(dt, u, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
        intervals = Simulation.find_continuous_intervals(find(gN_vec<=1e-5));

        figure
        subplot(5,1,1)
        timesteps = (0:size(x_b,2)-1)*dt;
        plot(timesteps, x_b, 'r', timesteps, x_p, 'b')
        Simulation.plot_intervals(intervals, dt)
        legend("Ball position [m]", "Plate position [m]")

        subplot(5,1,2)
        plot(timesteps, u_b, 'r', timesteps, u_p, 'b')
        Simulation.plot_intervals(intervals, dt)
        legend("Ball velocity [m/s]", "Plate velocity [m/s]")

        subplot(5,1,3)
        plot(timesteps, dP_N_vec)
        Simulation.plot_intervals(intervals, dt)
        legend("dP_N")

        subplot(5,1,4)
        plot(timesteps, gN_vec)
        Simulation.plot_intervals(intervals, dt)
        legend("g_{N_{vec}} [m]")

        subplot(5,1,5)
        plot(timesteps, u)
        Simulation.plot_intervals(intervals, dt)
        legend("F [N]")
    end

    function plot_intervals(intervals, dt, colors)
        if nargin<3 || length(colors) ~= size(intervals,2)
            colors = repelem({[91, 207, 244]/255},size(intervals,2));
        end
        j = 1;
        for i = intervals
            patch(dt*[i(1) i(1), i(2) i(2)], [min(ylim) max(ylim) max(ylim) min(ylim)], colors{j}, 'LineStyle', 'none', 'FaceAlpha', 0.3 )
            j=j+1;
        end
    end
   end
end
