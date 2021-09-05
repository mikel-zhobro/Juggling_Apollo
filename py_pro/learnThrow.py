## Feedforward controled system
# Here we want to set some convention to avoid missunderstandins later on.
# 1. the state is [xb, xp, ub, up]^T
# 2. the system can have as input either velocity u_des or the force F_p

# Design params
h_b_max = 1                # [m] maximal height the ball achievs
input_is_force = False

# INIT
# ----------------------------------------------------------------------- #
# Throw and catch point
[Tb, ub_00] = plan_ball_trajectory(h_b_max, 0, 0)   # flying time of the ball and required init velocity

# Init state
x_ruhe = -0.4
x0 = {x_ruhe x_ruhe 0 0} # the plate and ball in ruhe

# ILC
kf_d1d2_params.M_diag = 0.1                    # diagonal value of covarianc of noise on the measurment
kf_d1d2_params.P0_diag = 0.2                   # the diagonal value of initial variance of disturbance
kf_d1d2_params.epsilon0 = 0.3                  # initial variance of noise on the disturbance
kf_d1d2_params.epsilon_decrease_rate = 0.9     # the decreasing factor of noise on the disturbance

kf_dpn_params.M_diag = 0.1
kf_dpn_params.P0_diag = 0.1
kf_dpn_params.epsilon0 = 0.1
kf_dpn_params.epsilon_decrease_rate = 0.9

my_ilc = ILC('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'dt', dt,              ...
             'x_0', cell2mat(x0), 't_f', Tb,                      ...
             'kf_d1d2_params', kf_d1d2_params, 'kf_dpn_params', kf_dpn_params)

# my_ilc.resetILC()

# Iteration params
sim = Simulation('m_b', m_b, 'm_p', m_p, 'k_c', k_c, 'g', g, 'input_is_force', input_is_force, 'sys', my_ilc.sys, 'air_drag', True)
# ----------------------------------------------------------------------- #

## Learn Throw
close all
ILC_it = 1 # number of ILC iteration
ub_0 = ub_00
# reset ilc
[y_des,u_ff] = my_ilc.learnThrowStep(ub_00)
my_ilc.resetILC()

# Extra simulation to measure time of flight
T_sim_extra = 2*Tb
N_sim_extra = Simulation.steps_from_time(T_sim_extra, dt)

# collect: dup, x_p, x_b, u_p
dup_vec = zeros([ILC_it, size(my_ilc.kf_dpn.d)])
x_p_vec = zeros(ILC_it, my_ilc.N_1 + 1)
x_b_vec = zeros(ILC_it, N_sim_extra)
u_p_vec = zeros(ILC_it, my_ilc.N_1 + 1)
u_des_vec = zeros(ILC_it, my_ilc.N_1)
u_b0_vec = zeros(ILC_it, 1)
u_d2_vec = zeros(ILC_it, 1)
u_Tb_vec = zeros(ILC_it, 1)

# ILC Loop
# close all
# disturbance to be learned
period = 0.02/dt
disturbance = 200*sin(2*pi/period*(0:my_ilc.N_1-1)) # disturbance on the plate position
for j = 1:ILC_it
    display("ITERATION: " + num2str(j))

    # Main Simulation
    [x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, F_vec] = sim.simulate_one_iteration(dt, my_ilc.t_h/2, x0{:}, u_ff, 1,disturbance)

    # Measurments for height and fly-time-to-zero
    # Extra simulation to measure time of flight
    [x_b_extra, u_b_extra, x_p_extra, u_p_extra, dP_N_vec_extra, gN_vec_extra, F_vec_extra] = sim.simulate_one_iteration(dt, 2*Tb, x_b(end), 0, u_b(end), 0, zeros(N_sim_extra,1), 1)

    hb_meas = max(x_b)
    fly_time_meas = find(x_b_extra(2:end)<=1e-5, 1 , 'first')*dt

    # LEARN THROW
    #                                                   from 2 since we cant optimize the first state x(0)
    [y_des,u_ff, ub_0] = my_ilc.learnThrowStep(ub_0, u_ff, transpose(x_p(2:end)), hb_meas, fly_time_meas)

    # 5. Collect data for plotting
    dup_vec(j,:) =  my_ilc.kf_dpn.d
    x_p_vec(j,:) =  x_p
    x_b_vec(j,:) =  x_b_extra
    u_p_vec(j,:) =  u_p
    u_des_vec(j,:) =  u_ff
    u_d2_vec(j) =  my_ilc.kf_d1d2.d(2) #ub_0
    u_b0_vec(j) =  ub_0
    u_Tb_vec(j) = fly_time_meas
end
Simulation.plot_results(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
Simulation.plot_results(dt, F_vec_extra, x_b_extra, u_b_extra, x_p_extra, u_p_extra, dP_N_vec_extra, gN_vec_extra)

##
iter_steps = 1:ILC_it
tt = round(ILC_it/2)
figure plot(iter_steps,[u_Tb_vec, Tb*ones(ILC_it,1)]) legend("Tb through iterations", "Tb")
# figure plot(iter_steps,[u_Tb_vec - Tb*ones(ILC_it,1)]) legend("Tb through iterations", "Tb")
figure plot(iter_steps(tt:end),[u_b0_vec(tt:end)-u_b0_vec(end),u_d2_vec(tt:end), u_Tb_vec(tt:end)-Tb]) legend("ub0-ub0_{end} through iterations", "d2", "Tb_{meas}-Tb")
# Simulation.plot_results(dt, F_vec, x_b, u_b, x_p, u_p, dP_N_vec, gN_vec)
# Simulation.plot_results(dt, F_vec_extra, x_b_extra, u_b_extra, x_p_extra, u_p_extra, dP_N_vec_extra, gN_vec_extra)
## Plot trajectories over ILC iterations
close all
# plotIterations(dup_vec, "d_{p} through iterations", dt, 3)
# plotIterations(disturbance, "real disturbance", dt)
# plotIterations(x_b_vec, "x_b through iterations", dt, 3)
plotIterations(x_p_vec, "x_p through iterations", dt, 3)
plotIterations(u_p_vec, "u_p through iterations", dt, 2)
# plotIterations(u_des_vec, "u_{ff} through iterations", dt, 2)
