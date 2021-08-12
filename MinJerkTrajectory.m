classdef MinJerkTrajectory
    %MINJERKTRAJECTORY Used to do min-jerk trajectory computation
    % Since any free start- or end-state puts a constraint on the constate
    % the equations stay the same and only the coefficients change.
    % This allows us to call get_trajectories() to create paths of
    % different constraints.
    
   methods (Static)
    % X. Plate trajectory
    function [xp_des, T] = plan_plate_trajectory(dt, Tb, x_p0, x_pTb, up_0, up_Tb, ap_0, ap_T)
        % Input:
        %   xb_0, ub_0: conditions at t=0
        %   xb_Tb, ub_T: conditions at t=Tb  [assumed 0]
        % Output:
        %   xp_des(t) = [x_p(t);     u_p(t);       a_p(t);   u_(t)] 
        %             = [position; velocity; acceleration;    jerk]
        %   T is the total time for one cycle(i.e. throw and catch)

        % Get polynom parameters for different conditions
        if nargin < 6
            [c1, c2, c3, x3_0, x2_0, x1_0] = MinJerkTrajectory.free_start_end_acceleration(Tb, up_0, up_Tb, x_p0, x_pTb);
        elseif nargin < 7
            [c1, c2, c3, x3_0, x2_0, x1_0] = MinJerkTrajectory.free_end_acceleration(Tb, up_0, up_Tb, x_p0, x_pTb, ap_0);
        else
            [c1, c2, c3, x3_0, x2_0, x1_0] = MinJerkTrajectory.set_start_end_acceleration(Tb, up_0, up_Tb, x_p0, x_pTb, ap_0, ap_T);
        end

        % Calculate desired plate trajectories
        % 1. 0->Tb
        t = 0:dt:Tb;
        [u1, a_p1, u_p1, x_p1] = MinJerkTrajectory.get_trajectories(t, c1, c2, c3, x3_0, x2_0, x1_0);

        % 2. Tb->2Tb
        % TODO: maybe better call plan_plate_trajectory() again for this one
        tt = t(end)+dt-Tb:dt:Tb;
        tt = tt(end:-1:1);
        [u2, a_p2, u_p2, x_p2] = MinJerkTrajectory.get_trajectories(tt, c1, c2, c3, x3_0, x2_0, x1_0);

        % for t=Tb -> t=2Tb we just take the symmetric negative of the
        % corresponding trajectory in t=0 -> t=Tb
        % xp_des = [x_p1, -x_p2; u_p1, u_p1(end)-u_p2; a_p1, a_p1(end)-a_p2; u1, u1(end)-u2];
        xp_des = [x_p1, -x_p2; u_p1, u_p2; a_p1, -a_p2; u1, u2];
        T = tt(1) + Tb;
    end

    % Get function values from polynom parameters
    function [u, a_p, u_p, x_p] = get_trajectories(t, c1, c2, c3, x3_0, x2_0, x1_0)
        u   = -c1*t.^2 +    2*c2*t -     2*c3;
        a_p =  c1*t.^3/3  + c2*t.^2 +    2*c3*t +    x3_0;
        u_p =  c1*t.^4/12 + c2*t.^3/3 +  c3*t.^2 +   x3_0*t +       x2_0;
        x_p =  c1*t.^5/60 + c2*t.^4/12 + c3*t.^3/3 + x3_0*t.^2/2 +  t*x2_0 + x1_0;
    end

    % a). Acceleration is free at t=0 and t=Tb
    function [c1, c2, c3, x3_0, x2_0, x1_0] = free_start_end_acceleration(Tb, up_0, up_Tb, x_p0, x_pTb)

        C_cv = [5/(8*Tb^2), 15/(2*Tb^4),     -15/Tb^5;
                -3/(16*Tb), 15/(4*Tb^3), -15/(2*Tb^4);
                     Tb/96,   -7/(8*Tb),  15/(4*Tb^2)];

        v = [0;
             up_Tb-up_0;
             x_pTb-x_p0 - up_0*Tb];

        c = C_cv * v;

        c1 = c(1);
        c2 = c(2);
        x3_0 = c(3);
        c3 = 0;
        x2_0 = up_0;
        x1_0 = x_p0;
    end

    % b). Acceleration is set only at t=0
    function [c1, c2, c3, x3_0, x2_0, x1_0] = free_end_acceleration(Tb, ub_0, up_Tb, x_p0, x_pTb, ap_0)

        C_cv = [-10/(3*Tb^2), -60/Tb^4,  160/Tb^5;
                    4/(3*Tb),  36/Tb^3, -100/Tb^4;
                       -1/6,  -6/Tb^2,   20/Tb^3];

        v = [0; 
             up_Tb-ub_0 - Tb*ap_0; 
             x_pTb-x_p0 - ub_0*Tb - Tb^2*ap_0/2];

        c = C_cv * v;

        c1 = c(1);
        c2 = c(2);
        c3 = c(3);
        x3_0 = 0;
        x2_0 = ub_0;
        x1_0 = x_p0;
    end

    % c) Acceleration is set at t=0 and t=Tb
    function [c1, c2, c3, x3_0, x2_0, x1_0] = set_start_end_acceleration(Tb, up_0, up_Tb, x_p0, x_pTb, ap_0, ap_Tb)

        C_cv = [ 360/Tb^5, -180/Tb^4,  30/Tb^3;
                -180/Tb^4,   84/Tb^3, -12/Tb^2;
                  30/Tb^3,  -12/Tb^2, 3/(2*Tb)];

        v = [x_pTb-x_p0 - up_0*Tb - Tb^2*ap_0/2;
             up_Tb-up_0 - Tb*ap_0; 
             ap_Tb-ap_0]; % assume ap_t=ap_0

        c = C_cv * v;

        c1 = c(1);
        c2 = c(2);
        c3 = c(3);
        x3_0 = ap_0;
        x2_0 = up_0;
        x1_0 = x_p0;
    end

    function plot_paths(xp_des, T, dt, Title)
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
        sgtitle(Title)
    end
    
   end
end

