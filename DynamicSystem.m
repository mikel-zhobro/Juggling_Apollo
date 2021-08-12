classdef DynamicSystem < matlab.System
    %DYNAMICSYSEM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        dt;
        m_b; % mass of ball
        m_p; % mass of plate
        k_c; % force coefficient
        g;   % gravitational acceleration constant
    end
    
    methods
    % Constructor
    function obj = DynamicSystem(varargin)
        %OPTIMIZATIONDESIREDINPUT Construct an instance of this class
        %   Support name-value pair arguments when constructing object
        setProperties(obj,nargin,varargin{:})
    end
        
    function [Ad, Bd, Cd, S, x0, c] = getSystemMarixes(obj,dt)
        %METHOD1 Summary of this method goes here
        %   Detailed explanation goes here
        mbp = 1/(obj.m_p+obj.m_b);
        Ad = [ 1 0 dt/2*(2-obj.m_p*mbp)  dt/2*obj.m_p*mbp*(1-dt*obj.k_c)                            ; ...
               0 1 dt/2*obj.m_b*mbp      dt/2*(2 - obj.m_b*mbp(1 - dt*obj.k_c) - dt.k_c)            ; ...
               0 0 1-obj.m_p*mbp         obj.m_p*mbp*(1 - dt*obj.k_c)                               ; ...
               0 0 obj.m_b*mbp           -dt*obj.k_c - obj.m_b*mbp*(1- dt*obj.k_c)                  ];
           
       Bd = [dt^2/2*obj.m_p*obj.k_c*mbp         ; ...
             obj.k_c*dt^2/2*(1-obj.m_b*mbp)     ; ...
             obj.k_c*obj.m_p*dt*mbp             ; ...
             dt*obj.k_c*(1-obj.m_b*mbp)         ];
           
       
       c = [ -dt^2*obj.g/2*(1-obj.m_p*mbp)  ; ...
             -obj.m_b*dt^2*obj.g*mbp        ; ...
             -dt*obj.g*(1-obj.m_p*mbp)      ; ...
             -dt*obj.g*obj.m_b*mbp          ];
         
       Cd = [0 1 0 0];
       
       S =  [-dt/(2*obj.m_b)    ; ...
             dt/(2*obj.m_p)     ; ...
             -1/obj.m_b         ; ...
             1/obj.m_p          ];
       
         
    end
    end
end

