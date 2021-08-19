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
            u_des = quadprog((transpose(obj.lss.GF) * obj.lss.GF), transpose(obj.lss.GF)*(obj.lss.GK*dup + obj.lss.Gd0 - y_des));
            %% check
%             norm( (transpose(GF) * GF)* u_des + transpose(GF)*(GK*dup + Gd0 - y_des) )
        end
    end
end

