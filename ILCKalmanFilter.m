classdef ILCKalmanFilter < matlab.System
    % constants
    properties
        dt;
        n_d;
        I;
        M;
    end

    % initial values
    properties
        d0;
        P0;
        epsilon0;
    end

    % current values
    properties
        d;
        P;
        Omega; epsilon;
    end
    
    % set every iteration
    properties
        G;
        GF;
        Gd0;
    end

    methods
        
        function obj = ILCKalmanFilter(varargin)
            setProperties(obj,nargin,varargin{:})
            obj.n_d = length(obj.d0);
            obj.I = eye(obj.n_d);
        end
        
        function resetKF(obj)
            obj.d = obj.d0;
            obj.P = obj.P0;
            obj.Omega = eye(obj.n_d) * obj.epsilon0;
            obj.epsilon = obj.epsilon0;
        end

        function set_G_GF_Gd0(obj, G, GF, Gd0)
            obj.G = G;
            obj.GF = GF;
            obj.Gd0 = Gd0;
        end

        % Disturbance is on the state trajectory
        function d = updateStep(obj, u, y)
            % In this case y = Fu + Gd0  + [Gd] + mu
            P1_0 = obj.P + obj.Omega;
            Theta = obj.G*P1_0*transpose(obj.G) + obj.M;
            K = P1_0*transpose(obj.G)*Theta^-1;
            obj.P = (obj.I - K*obj.G)* P1_0;
            obj.d = obj.d + K * ( y - obj.Gd0 - obj.G*obj.d - obj.GF*u);
            
            % update epsilon
            obj.epsilon = obj.epsilon*0.9;
            d = obj.d;
        end
        
        % Disturbance is on the output trajectory
        function d = updateStep2(obj, u, y)
            % In this case y = Fu + Gd0  + [d] + mu, so G=I
            P1_0 = obj.P + obj.Omega;                               % Predicted a priori covariance estimate
            Theta = P1_0 + obj.M;                                   % Inovation covariance
            K = P1_0*Theta^-1;                                      % Optimal Kalman Gain
            obj.P = (obj.I - K)* P1_0;                              % Update a posteriori covariance estimate
            obj.d = obj.d + K * ( y - obj.Gd0 - obj.d - obj.GF*u);  % Update a posteriori satte estimate
            d = obj.d;
            % update epsilon
            obj.epsilon = obj.epsilon*0.8;
        end
    end
end

