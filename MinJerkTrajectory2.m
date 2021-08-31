classdef MinJerkTrajectory2
    %MINJERKTRAJECTORY2 Used to do min-jerk trajectory computation
    % Since any free start- or end-state puts a constraint on the constate
    % the equations stay the same and only the coefficients change.
    % This allows us to call get_trajectories() to create paths of
    % different constraints.
    
   methods (Static)
    function [x, v, a, j] = get_min_jerk_trajectory(dt, ta, tb, x_ta, x_tb, u_ta, u_tb, a)
        % Input:
        %   x_ta, u_ta, (optional: a.ta): conditions at t=ta
        %   x_tb, u_tb, (optional: a.tb): conditions at t=tb
        %   a: is set to [] if start and end acceleration are free
        % Output:
        %   xp_des(t) = [x(t);       u(t);         a(t);            u(t)]
        %             = [position;   velocity;     acceleration;    jerk]
        
        % Get polynom parameters for different conditions
        T = tb-ta;
        if nargin>7
            if isfield(a, 'ta')
                % 1. set start acceleration
                if isfield(a, 'tb')
                    % a. set end acceleration
                    [c1, c2, c3, c4, c5, c6] = MinJerkTrajectory2.set_start_acceleration(T, x_ta, x_tb, u_ta, u_tb, a.ta, a.tb);
                else
                    % b.free end acceleration
                    [c1, c2, c3, c4, c5, c6] = MinJerkTrajectory2.set_start_acceleration(T, x_ta, x_tb, u_ta, u_tb, a.ta);
                end
            else 
                % 2. free start acceleration
                if isfield(a, 'tb')
                    % a. set end acceleration
                    [c1, c2, c3, c4, c5, c6] = MinJerkTrajectory2.free_start_acceleration(T, x_ta, x_tb, u_ta, u_tb, a.tb);
                else
                    % b.free end acceleration
                    [c1, c2, c3, c4, c5, c6] = MinJerkTrajectory2.free_start_acceleration(T, x_ta, x_tb, u_ta, u_tb);
                end
            end
        else
            % free start&end acceleration
            [c1, c2, c3, c4, c5, c6] = MinJerkTrajectory2.free_start_acceleration(T, x_ta, x_tb, u_ta, u_tb);
        end
        
        % Trajectory values ta->tb
        t = 0:dt:T;
        [j, a, v, x] = MinJerkTrajectory2.get_trajectories(t, c1, c2, c3, c4, c5, c6);
    end

    % Get function values from polynom parameters
    function [j, a, v, x] = get_trajectories(t, c1, c2, c3, c4, c5, c6)
        j =  c1*t.^2/2   - c2*t       + c3;                               % jerk
        a =  c1*t.^3/6   - c2*t.^2/2  + c3*t      + c4;                    % acceleration
        v =  c1*t.^4/24  - c2*t.^3/6  + c3*t.^2/2 + c4*t      + c5;        % velocity
        x =  c1*t.^5/120 - c2*t.^4/24 + c3*t.^3/6 + c4*t.^2/2 + c5*t + c6; % position
    end

    % 1) Acceleration is set at t=0 (a(0)=a0 => c4=a0)
    function [c1, c2, c3, c4, c5, c6] = set_start_acceleration(T, x0, xT, u0, uT, a0, aT)
        if nargin<7
            % free end acceleration u(T)=0
            M = [320/T^5, -120/T^4, -20/(3*T^2); 200/T^4, -72/T^3, -8/(3*T); 40/T^3, -12/T^2, -1/3];
            c = [-(a0*T^2)/2 - u0*T - x0 + xT; uT - u0 - T*a0; 0];  
        else
            % set end acceleration a(T)=aT
            M = [720/T^5, -360/T^4, 60/T^3; 360/T^4, -168/T^3, 24/T^2; 60/T^3, -24/T^2, 3/T];
            c = [-(a0*T^2)/2 - u0*T - x0 + xT; uT - u0 - T*a0; aT - a0];
        end
        
        c123 = M*c;
        c1 = c123(1);
        c2 = c123(2);
        c3 = c123(3);
        c4 = a0;
        c5 = u0;
        c6 = x0;
    end

    % 2) Acceleration is free at t=0 (u(0)=0 => c3=0)
    function [c1, c2, c3, c4, c5, c6] = free_start_acceleration(T, x0, xT, u0, uT, aT)
        if nargin<7
            % free end acceleration u(T)=0
            M = [120/T^5, -60/T^4, -5/T^2; 60/T^4, -30/T^3, -3/(2*T); 5/T^2, -3/(2*T), -T/24];
            c = [xT - x0 - T*u0; uT - u0; 0];  
        else
            % set end acceleration a(T)=aT
            M = [320/T^5, -200/T^4, 40/T^3; 120/T^4, -72/T^3, 12/T^2; 20/(3*T^2), -8/(3*T), 1/3];
            c = [xT - x0 - T*u0; uT - u0; aT];
        end
        
        c123 = M*c;
        c1 = c123(1);
        c2 = c123(2);
        c4 = c123(3);
        c3 = 0;
        c5 = u0;
        c6 = x0;
    end

    function plot_paths(x, v, a, j, dt, Title, intervals, colors)
        if nargin < 8
            colors = [];
        end
        figure
        timesteps = (1:length(x))*dt;
        subplot(4,1,1)
        plot(timesteps, x)
        Simulation.plot_intervals(intervals, dt, colors)
        
        legend("Plate position")

        subplot(4,1,2)
        plot(timesteps, v)
        Simulation.plot_intervals(intervals, dt, colors)
        legend("Plate velocity")

        subplot(4,1,3)
        plot(timesteps, a)
        Simulation.plot_intervals(intervals, dt, colors)
        legend("Plate acceleration")

        subplot(4,1,4)
        plot(timesteps, j)
        Simulation.plot_intervals(intervals, dt, colors)
        legend("Plate jerk")
        sgtitle(Title)
    end
    
   end
end

