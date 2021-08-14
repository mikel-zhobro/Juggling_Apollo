classdef OptimizationDesiredInput < matlab.System
    %OPTIMIZATIONDESIREDINPUT Summary of this class goes here
    %   Detailed explanation goes here

    properties
        Ad_Ad_impact;
        Bd_Bd_impact;
        c_c_impact;
        Ad; Ad_impact;  % [nx, nx]
        Bd; Bd_impact;  % [nx, nu]
        Cd;             % [ny, nx]
        S;              % [nx, ndup]
        x0;             % initial state (xb0, xp0, ub0, up0)
        c;  c_impact;   % constants from gravity ~ dt, g, mp mb
    end
%     properties (Access = private)
%        F;       % [nx*N, nu*N]
%        G;       % [ny*N, nx*N]
%        K;       % [nx*N, ndup*N]
%        d0;      % [nx*N, 1]
%        GF;      % helper
%        GFTGF_1; % ((G*F)^T * (G*F))^-1
%        GK;      % G * k;
%        Gd0;     % G * d0
%     end

    methods
        % Constructor
        function obj = OptimizationDesiredInput(varargin)
            setProperties(obj,nargin,varargin{:})
            obj.Ad_Ad_impact = {obj.Ad, obj.Ad_impact}; % Ad_Ad_impact{1} == Ad, Ad_Ad_impact{2} == Ad_impact
            obj.Bd_Bd_impact = {obj.Bd, obj.Bd_impact};
            obj.c_c_impact = {obj.c, obj.c_impact};
        end

        function [GF, GK, Gd0] = calcQuadrProgMatrixes(obj, set_of_impact_timesteps)
            % sizes
            N = length(set_of_impact_timesteps);    % nr of steps
            nx = size(obj.Ad,1);
            ny = size(obj.Cd,1);
            nu = size(obj.Bd,2);
            ndup =  size(obj.S,2);

            % set_of_impact_timesteps{t} = 1 if no impact, = 2 if impact.
            A_power_holder = cell(1, N+1);
            A_power_holder{1} = ones(size(obj.Ad));
            for i=1:N
                A_power_holder{i+1} = obj.Ad_Ad_impact{set_of_impact_timesteps(i)} * A_power_holder{i};
            end

            % Create lifted-space matrixes
            % Create matrixes F, K, G, d0
            d0 = transpose(cell2mat(cellfun(@(A){transpose(A*obj.x0)}, A_power_holder(2:end))));
            c_vec = transpose(cell2mat(arrayfun(@(ii){transpose(obj.c_c_impact{ii})}, set_of_impact_timesteps)));
            F = zeros(nx*N, nu*N);
            K = zeros(nx*N, ndup*N);
            G = zeros(ny*N, nx*N);
            M = zeros(nx*N, nx*N);
            for l=1:N
                G((l-1)*ny+1:l*ny,(l-1)*nx+1:l*nx) = obj.Cd;
                for m=1:l
                    M((l-1)*nx+1:l*nx,(m-1)*nx+1:m*nx) = A_power_holder{l-m+1};
                    F((l-1)*nx+1:l*nx,(m-1)*nu+1:m*nu) = A_power_holder{l-m+1}*obj.Bd_Bd_impact{set_of_impact_timesteps(m)};  % F_lm
                    K((l-1)*nx+1:l*nx,(m-1)*ndup+1:m*ndup) = A_power_holder{l-m+1}*obj.S;
                end
            end
            d0 = d0 + M * c_vec;

            % Prepare matrixes needed for the quadratic problem
            GF = G*F;
            % GFTGF = (transpose(GF) * (GF))^-1;
            GK = G * K;
            Gd0 = G * d0;


        end

        function u_des = calcDesiredInput(obj, dup, y_des, set_of_impact_timesteps)
            [GF, GK, Gd0] = obj.calcQuadrProgMatrixes(set_of_impact_timesteps);
            u_des = linsolve((transpose(GF) * GF), transpose(GF)*(GK*dup + Gd0 - y_des));
        end

        function u_des = calcDesiredInput2(obj, dup, y_des) % give as input Tb, or N_Tb(from which index Tb starts)
            %CALCDESIREDINPUT Once we have the desired motion y_des for the
            %plate and have estimated disturbances we can compute the
            %"optimal" feedforward v_des by solving the QP problem.
            %TODO: maybe constrained QP (to make sure state doesnt gocrazy)
            %   dup: [ndup*(N+1),1]  [dup_0,..dip_(N-1)]
            %   ydes: [ndy*(N+1),1]   [ydes_0,..ydes_N]
            u_des = linsolve((transpose(obj.GF) * obj.GF), transpose(obj.GF)*(obj.GK*dup + obj.Gd0 - y_des));
            % u_des = -obj.GFTGF_1 * (obj.GK*dup + obj.Gd0 - y_des)*obj.GF;
        end
    end
end

