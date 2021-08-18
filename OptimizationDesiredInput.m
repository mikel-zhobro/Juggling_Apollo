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
    properties
%        F;       % [nx*N, nu*N]
       G;       % [ny*N, nx*N]
%        K;       % [nx*N, ndup*N]
%        d0;      % [nx*N, 1]
       GF;      % helper
%        GFTGF_1; % ((G*F)^T * (G*F))^-1
       GK;      % G * k;
       Gd0;     % G * d0
    end

    methods
        % Constructor
        function obj = OptimizationDesiredInput(varargin)
            setProperties(obj,nargin,varargin{:})
            obj.Ad_Ad_impact = {obj.Ad, obj.Ad_impact}; % Ad_Ad_impact{1} == Ad, Ad_Ad_impact{2} == Ad_impact
            obj.Bd_Bd_impact = {obj.Bd, obj.Bd_impact};
            obj.c_c_impact = {obj.c, obj.c_impact};
        end

        function updateQuadrProgMatrixes(obj, set_of_impact_timesteps)
            % set_of_impact_timesteps{t} = 1 if no impact, = 2 if impact for the timesteps 0 -> N-1
            % sizes
            N = length(set_of_impact_timesteps);    % nr of steps
            nx = size(obj.Ad,1);
            ny = size(obj.Cd,1);
            nu = size(obj.Bd,2);
            ndup = size(obj.S, 2);

            % calculate I, A_1, A_2*A_1, .., A_N-1*A_N-2*..*A_1
            A_power_holder = cell(1, N);
            A_power_holder{1} = eye(size(obj.Ad));
            for i=1:N-1
                A_power_holder{i+1} = obj.Ad_Ad_impact{set_of_impact_timesteps(i+1)} * A_power_holder{i};
            end

            % Create lifted-space matrixes
            % Create matrixes F, K, G, M: x = Fu + Kdu_p + d0, y = Gx, 
            % where the constant part d0 = L*x0_N-1 + M*c0_N-1
            % F = [B0          0        0  .. 0
            %      A1B0        B1       0  .. 0
            %      A2A1B0      A1B0     B0 .. 0
            %        ..         ..         ..
            %      AN-1..A1B0  AN-2..A1B0  .. B0]
            F = zeros(nx*N, nu*N);
            % ---------- uncomment if dup is disturbance on dPN -----------
            % ndup =  size(obj.S,2);
            % K = [S          0       0 .. 0
            %      A1S        S       0 .. 0
            %      A2A1S      A1S     S .. 0
            %        ..       ..        ..
            %      AN-1..A1S AN-2..A1S  .. S]
            K = zeros(nx*N, ndup*N);
            % -------------------------------------------------------------
            % G = [Cd 0  .. .. 0
            %      0  Cd 0  .. 0
            %      .. .. .. .. ..
            %      0  0  0  .. Cd]
            obj.G = zeros(ny*N, nx*N);
            % M = [I         0      0 .. 0
            %      A1        I      0 .. 0
            %      A2A1      A1     I .. 0
            %       ..       ..       ..
            %      AN-1..A1 AN-2..A1  .. I]
            M = zeros(nx*N, nx*N);
            % L = [A0 0     ..        0
            %      0  A1A0  ..        0
            %      ..  ..   ..       ...
            %      0   0    ..   AN-1AN-2..A0]
            L = zeros(nx*N, nx*N);
            A_0 = obj.Ad_Ad_impact{set_of_impact_timesteps(1)};
            for l=1:N
                obj.G((l-1)*ny+1:l*ny,(l-1)*nx+1:l*nx) = obj.Cd;
                L((l-1)*nx+1:l*nx,(l-1)*nx+1:l*nx) = A_power_holder{l}*A_0;
                for m=1:l
                    M((l-1)*nx+1:l*nx,(m-1)*nx+1:m*nx) = A_power_holder{l-m+1};
                    F((l-1)*nx+1:l*nx,(m-1)*nu+1:m*nu) = A_power_holder{l-m+1}*obj.Bd_Bd_impact{set_of_impact_timesteps(m)};  % F_lm
                    % ------ uncomment if dup is disturbance on dPN -------
                    K((l-1)*nx+1:l*nx,(m-1)*ndup+1:m*ndup) = A_power_holder{l-m+1}*obj.S;
                    % -----------------------------------------------------
                end
            end
            % Create d0 = L*x0_N-1 + M*c0_N-1
            % d0 = sum_{n=0_N-1}(prod_{i=0_n} A_i*x0  , i.e  A_0*x0 + A_1*A_0*x0 ... A_{N-1}*..*A_0*x0 <- the x0 depending part of d0
%             A_0 = obj.Ad_Ad_impact{set_of_impact_timesteps(1)};
%             d0 = transpose(cell2mat(cellfun(@(A_n){transpose(A_n*A_0*obj.x0)}, A_power_holder)));
%             d0 = d0 + M * c_vec;
            c_vec = transpose(cell2mat(arrayfun(@(ii){transpose(obj.c_c_impact{ii})}, set_of_impact_timesteps)));
            d0 = L * repmat(obj.x0, [N,1]) + M * c_vec;

            % Prepare matrixes needed for the quadratic problem
            obj.GF = obj.G*F;
            % ---------- uncomment if dup is disturbance on dPN -----------
            obj.GK = obj.G * K;
%             GK = G; % if dup is disturbance on the state trajectory
%             GK = 1; % if dup is disturbance on the output trajectory
            % -------------------------------------------------------------
            obj.Gd0 = obj.G * d0;
            
            % save matrixes
        end

        function u_des = calcDesiredInput(obj, dup, y_des, set_of_impact_timesteps)
%             u_des = linsolve(GF, -transpose(GF)*(GK*dup + Gd0 - y_des));
            u_des = quadprog((transpose(obj.GF) * obj.GF), transpose(obj.GF)*(obj.GK*dup + obj.Gd0 - y_des));
            %% check
%             norm( (transpose(GF) * GF)* u_des + transpose(GF)*(GK*dup + Gd0 - y_des) )
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

