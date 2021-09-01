classdef OptimizationDesiredInput < matlab.System
    properties
    lss;
    end

    methods
        % Constructor
        function obj = OptimizationDesiredInput(lifted_state_space)
            obj.lss = lifted_state_space;
        end

        function u_des = calcDesiredInput(obj, dup, y_des)
%             u_des = linsolve(GF, -transpose(GF)*(GK*dup + Gd0 - y_des));
            % Add penalty on the input
            u_des = quadprog((transpose(obj.lss.GF) * obj.lss.GF), transpose(obj.lss.GF)*(obj.lss.GK*dup + obj.lss.Gd0 - y_des));
            %% check
%             norm( (transpose(obj.lss.GF) * obj.lss.GF)* u_des + transpose(obj.lss.GF)*(obj.lss.GK*dup + obj.lss.Gd0 - y_des) )
        end
    end
end

