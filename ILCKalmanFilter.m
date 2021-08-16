classdef ILCKalmanFilter < matlab.System
    %ILCKALMANFILTER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n_d;
        d;
        P;
        dt;
        Omega; epsilon;
        M;
        
        G;
        F;
        GF;
        I;
        
    end
    
    methods

        function obj = ILCKalmanFilter(varargin)
            % Support name-value pair arguments when constructing object
            setProperties(obj,nargin,varargin{:})
            obj.n_d = length(obj.d);
            obj.Omega = eye(obj.n_d) * obj.epsilon;
            obj.I = eye(obj.n_d);
        end
        
        function set_G_GF(obj, G, GF)
            obj.G =G;
            obj.GF = GF;
        end

        function d = updateStep(obj, u, y)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            P1_0 = obj.P + obj.Omega;
            Theta = obj.G*P1_0*transpose(obj.G) + obj.M;
            K = P1_0*transpose(obj.G)*Theta^-1;
            obj.P = (obj.I - K*obj.G)* P1_0;
            obj.d = obj.d + K * ( y - obj.G*obj.d - obj.GF*u);
            
            % update epsilon
            obj.epsilon = obj.epsilon*0.9;
            
            d = obj.d;
        end
    end
end

