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
        lss;
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
        
        function d = updateStep(obj, u, y)
            % In this case y = Fu + Gd0  + ((( GKd ))) + mu
            P1_0 = obj.P + obj.Omega;
            Theta = obj.lss.GK * P1_0 * transpose( obj.lss.GK ) + obj.M;
            K = P1_0 * transpose( obj.lss.GK ) * Theta^-1;
            obj.P = ( obj.I - K*obj.lss.GK ) * P1_0;
            obj.d = obj.d + K * ( y - obj.lss.Gd0 - obj.lss.GK*obj.d - obj.lss.GF*u );
            
            % update epsilon
            obj.epsilon = obj.epsilon*0.9;
            d = obj.d;
        end
    end
end

