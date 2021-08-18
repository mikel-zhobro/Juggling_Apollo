classdef OptimizationDesiredInput < matlab.System
    properties
    lss;
    end

    methods
        % Constructor
        function obj = OptimizationDesiredInput(lifted_state_space)
            obj.lss = lifted_state_space;
        end

        function u_des = calcDesiredInput(obj, dup, y_des) % give as input Tb, or N_Tb(from which index Tb starts)
            %CALCDESIREDINPUT Once we have the desired motion y_des for the
            %plate and have estimated disturbances we can compute the
            %"optimal" feedforward v_des by solving the QP problem.
            %TODO: maybe constrained QP (to make sure state doesnt gocrazy)
            %   dup: [ndup*(N+1),1]  [dup_0,..dip_(N-1)]
            %   ydes: [ndy*(N+1),1]   [ydes_0,..ydes_N]
            u_des = linsolve((transpose(obj.lss.GF) * obj.lss.GF), transpose(obj.lss.GF)*(obj.lss.GK*dup + obj.lss.Gd0 - y_des));
            % u_des = -obj.GFTGF_1 * (obj.GK*dup + obj.Gd0 - y_des)*obj.GF;
        end
    end
end

