classdef Simulation < matlab.System
    %SIMULATION Summary of this class goes here
    %   Detailed explanation goes here
    properties
        m_b; % mass of ball
        m_p; % mass of plate
        k_c; % force coefficient
        g;   % gravitational acceleration constant
    end

   methods
    % Constructor
    function obj = Simulation(varargin)
        %OPTIMIZATIONDESIREDINPUT Construct an instance of this class
        %   Support name-value pair arguments when constructing object
        setProperties(obj,nargin,varargin{:})
    end

    function [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec] = simulate_one_iteration(obj, dt, T, x_b0, x_p0, u_b0, u_p0, u, input_is_force)
        % if otherwise not specified input is force
        if nargin<9
          input_is_force=true;
        end

        % Initialize state vectors of the system
        N = ceil(T/dt);
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
                F_p = obj.force_from_velocity(u(i), u_p(i));
            end
            [x_b_new, x_p_new, u_b_new, u_p_new, dP_N, gN] = obj.simulate_one_step(dt, F_p, x_b(i), x_p(i), u_b(i), u_p(i));

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

    function F_p = force_from_velocity(obj, u_des_p, u_p)
        F_p  = obj.m_p * obj.k_c * (u_des_p-u_p);
    end

    function [x_b_new, x_p_new, u_b_new, u_p_new, dP_N, gN] = simulate_one_step(obj, dt, F_p_i, x_b_i, x_p_i, u_b_i, u_p_i)
        x_b_1_2 = x_b_i + 0.5*dt*u_b_i;
        x_p_1_2 = x_p_i + 0.5*dt*u_p_i;

        gN = x_b_1_2 - x_p_1_2;
        gamma_n_i = u_b_i - u_p_i;
        if gN <=0
            dP_N = max(0,(-gamma_n_i + obj.g*dt + F_p_i*dt/obj.m_p)/ (obj.m_b^-1 + obj.m_p^-1));
        else
            dP_N = 0;
        end

        u_b_new = u_b_i - obj.g*dt + dP_N/obj.m_b;
        u_p_new = u_p_i + F_p_i*dt/obj.m_p - dP_N/obj.m_p;

        x_b_new = x_b_1_2 + u_b_new*dt/2;
        x_p_new = x_p_1_2 + u_p_new*dt/2;
    end
   end
   
   %% Static Helpers
   methods (Static)
       
    function N = steps_from_time(T, dt)
        N = ceil(T/dt);
    end
    
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
    
    function plot_results(dt, F_p, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
        intervals = Simulation.find_continuous_intervals(find(gN_vec<0));
        
        figure
        subplot(5,1,1)
        timesteps = [1:size(x_b,2)]*dt;
        plot(timesteps, x_b, 'r', timesteps, x_p, 'b')
        Simulation.plot_intervals(intervals, dt)
        legend("Ball position", "Plate position")

        subplot(5,1,2)
        plot(timesteps, u_b, 'r', timesteps, u_p, 'b')
        Simulation.plot_intervals(intervals, dt)
        legend("Ball velocity", "Plate velocity")

        subplot(5,1,3)
        plot(timesteps, dP_N_vec)
        Simulation.plot_intervals(intervals, dt)
        legend("dP_N")

        subplot(5,1,4)
        plot(timesteps, gN_vec)
        Simulation.plot_intervals(intervals, dt)
        legend("g_{N_{vec}}")

        subplot(5,1,5)
        plot(timesteps, F_p)
        Simulation.plot_intervals(intervals, dt)
        legend("F_p")
    end
    
    function plot_intervals(intervals, dt)
        for i = intervals
            patch(dt*[i(1) i(1), i(2) i(2)], [min(ylim) max(ylim) max(ylim) min(ylim)], [91, 207, 244]/255, 'LineStyle', 'none', 'FaceAlpha', 0.3 )
        end
    end
   end
end