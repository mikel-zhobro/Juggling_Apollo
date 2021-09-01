classdef DynamicSystem < matlab.System

    properties
        m_b; % mass of ball
        m_p; % mass of plate
        k_c; % force coefficient
        g;   % gravitational acceleration constant
        dt;
    end

    methods
        % Constructor
        function obj = DynamicSystem(varargin)
            setProperties(obj,nargin,varargin{:})
        end

        function [Ad, Bd, Cd, S, c] = getSystemMarixesVelocityControl(obj,dt,contact_impact)
            % x_k = Ad*x_k-1 + Bd*u_k-1 + S*d_k + c
            % y_k = Cd*x_k
            if nargin<3
                contact_impact=false;
            end

            if contact_impact
                mbp = 1/(obj.m_p+obj.m_b);
            else
                mbp = 0;
            end
            dt_2 = 0.5 * dt;
            Ad = [ 1 0 dt-dt_2*obj.m_p*mbp   dt_2*obj.m_p*mbp*(1 - dt*obj.k_c)                    ; ...
                   0 1 dt_2*obj.m_b*mbp      dt_2*(2 - dt*obj.k_c - obj.m_b*mbp*(1 - dt*obj.k_c)) ; ...
                   0 0 1-obj.m_p*mbp         obj.m_p*mbp*(1 - dt*obj.k_c)                         ; ...
                   0 0 obj.m_b*mbp           1 - dt*obj.k_c - obj.m_b*mbp*(1 - dt*obj.k_c)        ];

            Bd = [dt_2*dt*mbp*obj.m_p*obj.k_c       ; ...
                  obj.k_c*dt_2*dt*(1-obj.m_b*mbp)   ; ...
                  dt*mbp*obj.m_p*obj.k_c            ; ...
                  dt*obj.k_c*(1-obj.m_b*mbp)        ];


            c = [-dt_2*dt*obj.g*(1-obj.m_p*mbp) ; ...
                 -dt_2*dt*obj.g*obj.m_b*mbp     ; ...
                 -dt*obj.g*(1-obj.m_p*mbp)      ; ...
                 -dt*obj.g*obj.m_b*mbp          ];

            Cd = [0 1 0 0];
%             Cd = [0 1 0 0; ...
%                   0 0 0 1];

            S =  [-dt_2/obj.m_b ; ...
                  dt_2/obj.m_p  ; ...
                  -1/obj.m_b    ; ...
                  1/obj.m_p     ];
        end

        function [Ad, Bd, Cd, S, c] = getSystemMarixesForceControl(obj,dt,contact_impact)
            % x_k = Ad*x_k-1 + Bd*u_k-1 + S*d_k + c
            % y_k = Cd*x_k
            if nargin<3
                contact_impact=false;
            end

            if contact_impact
                mbp = 1/(obj.m_p+obj.m_b);
            else
                mbp = 0;
            end
            dt_2 = 0.5 * dt;
                 % xb xp ub up
            Ad = [ 1 0 dt-dt_2*obj.m_p*mbp   dt_2*obj.m_p*mbp       ; ...
                   0 1 dt_2*obj.m_b*mbp      dt_2*(2 - obj.m_b*mbp)   ; ...
                   0 0 1-obj.m_p*mbp         obj.m_p*mbp            ; ...
                   0 0 obj.m_b*mbp           1-obj.m_b*mbp          ];

            Bd = [dt_2*dt*mbp                       ; ...
                 1/obj.m_p*dt_2*dt*(1-obj.m_b*mbp)  ; ...
                 dt*mbp                             ; ...
                 dt/obj.m_p*(1-obj.m_b*mbp)         ];


            c = [-dt_2*dt*obj.g*(1-obj.m_p*mbp) ; ...
                 -dt_2*dt*obj.g*obj.m_b*mbp     ; ...
                 -dt*obj.g*(1-obj.m_p*mbp)      ; ...
                 -dt*obj.g*obj.m_b*mbp          ];

            Cd = [0 1 0 0];
%             Cd = [0 1 0 0; ...
%                   0 0 0 1];

            S =  [-dt_2/obj.m_b ; ...
                  dt_2/obj.m_p  ; ...
                  -1/obj.m_b    ; ...
                  1/obj.m_p     ];

    end
    end
end

