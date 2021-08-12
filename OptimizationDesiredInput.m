classdef OptimizationDesiredInput < matlab.System
    %OPTIMIZATIONDESIREDINPUT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Ad; % [nx, nx] 
        Bd; % [nx, nu]
        Cd; % [ny, nx]
        S;  % [nx, ndup]
        x0; % initial state (xb0, xp0, ub0, up0)
        c;  % constants from gravity ~ dt, g, mp mb
        N;  % nr of steps
    end
    properties (Access = private)
       F;       % [nx*N, nu*N] 
       G;       % [ny*N, nx*N] 
       K;       % [nx*N, ndup*N] 
       d0;      % [nx*N, 1]
       GF;  % helper
       GFTGF_1; % ((G*F)^T * (G*F))^-1
       GK;      % G * k;
       Gd0;     % G * d0
    end
    
    methods
        % Constructor
        function obj = OptimizationDesiredInput(varargin)
            %OPTIMIZATIONDESIREDINPUT Construct an instance of this class
            %   Support name-value pair arguments when constructing object
            setProperties(obj,nargin,varargin{:})
            obj.initLiftedSpace()
            display(obj.GFTGF_1)
            display(obj.GK)
            display(obj.Gd0)
        end
        function obj = initLiftedSpace(obj)
            nx = size(obj.Ad,1);
            ny = size(obj.Cd,1);
            nu = size(obj.Bd,2);
            ndup =  size(obj.S,2);
                      
            %% Pre-calculate powers of Ad using its eigenvalue decomposition
            % Calculate eigenvalues decomposition of Ad
            [V,D,W] = eig(obj.Ad);
            eigens = diag(D);
            
            % Collect all powers of eigenvalues
            eigen_holder = cell(1, obj.N+1);
            eigen_holder{1} = ones(size(eigens));
            eigen_holder{2} = eigens;
            for i = 3:obj.N+1
               eigen_holder{i} = eigen_holder{i-1} .* eigens; % correct
            end
            % collect all powers of Ad: Ad^n for n=1,..,N
            A_power_holder = cellfun(@(n){V * diag(n) * W}, eigen_holder);  % [I, A, A^2,..A^N] correct N+1
                 
            %% Create lifted-space matrixes
            % Create matrixes F, K, G, d0
            obj.d0 = transpose(cell2mat(cellfun(@(A){transpose(A*obj.x0)}, A_power_holder(2:end))));
            obj.F = zeros(nx*obj.N, nu*obj.N);
            obj.K = zeros(nx*obj.N, ndup*obj.N);
            obj.G = zeros(ny*obj.N, nx*obj.N);
            M = zeros(nx*obj.N, nx*obj.N);
            for l=1:obj.N
                obj.G((l-1)*ny+1:l*ny,(l-1)*nx+1:l*nx) = obj.Cd;
                for m=1:l
                    M((l-1)*nx+1:l*nx,(m-1)*nx+1:m*nx) = A_power_holder{l-m+1};
                    obj.F((l-1)*nx+1:l*nx,(m-1)*nu+1:m*nu) = A_power_holder{l-m+1}*obj.Bd;
                    obj.K((l-1)*nx+1:l*nx,(m-1)*ndup+1:m*ndup) = A_power_holder{l-m+1}*obj.S;
                end
            end
            obj.d0 = obj.d0 + M * repmat(obj.c,obj.N,1);
            
            % Prepare matrixes needed for the quadratic problem
            obj.GF = obj.G*obj.F;
            obj.GFTGF_1 = (transpose(obj.GF) * (obj.GF))^-1;
            obj.GK = obj.G * obj.K;
            obj.Gd0 = obj.G * obj.d0;
        end
       
        function u_des = calcDesiredInput(obj, dup, y_des)
            %METHOD1 Calculate deired input from desired trajectory
            %        and estimated disturbances.
            %   dup: [ndup*(N+1),1]  [dup_0,..dip_(N-1)]
            %   ydes: [ndy*(N+1),1]   [ydes_0,..ydes_N]
            u_des = linsolve((transpose(obj.GF) * obj.GF), transpose(obj.GF)*(obj.GK*dup + obj.Gd0 - y_des));
            % u_des = -obj.GFTGF_1 * (obj.GK*dup + obj.Gd0 - y_des)*obj.GF;
        end
    end
end

