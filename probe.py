import matplotlib.pyplot as plt
import numpy as np
from apollo_interface.Apollo_It import ApolloInterface
from juggling_apollo.settings import alpha
from math import sin

def main():
    N = 2000
    dt = 0.004
    timesteps = np.arange(0.0, dt*N,dt)
    inputs = np.zeros((N, 7))
    inputs[:,0] = 0.3 * np.sin(timesteps)
    
    r_arm = ApolloInterface(r_arm=True)
    r_arm.go_to_posture_array([0.0, 0.0, -np.pi/4, np.pi/2, np.pi/2, np.pi/2, 0.0], 2000, False)
    # r_arm.go_to_posture_array([np.pi/4, 0.0, np.pi/4, np.pi/4, 3*np.pi/4, 3*np.pi/4, 0.0], 2000, False)
    
    poses, velocities, acc, _, u_vec = r_arm.apollo_run_one_iteration(dt, T=dt*len(timesteps), u=inputs)
    
    plt.figure()
    plt.plot(timesteps, poses[:, 0], label='angle')
    plt.plot(timesteps, velocities[:, 0], label='velocity')
    plt.plot(timesteps, u_vec[:, 0], label='des_velocity')
    plt.plot(timesteps, acc[:, 0], label='acc')
    plt.legend()
    plt.show()
    

def main2():

       # define the model and draw some data
    model = lambda x: x * np.sin(x)
    xdata = np.arange(0, 12, 0.2)
    ydata = model(xdata)

    plt.plot(xdata, ydata, '-', color='k', label="psi")
    plt.fill_between(xdata, ydata -2, ydata + 2, color='gray', alpha=0.2, label="psimin psimax")
    plt.legend()
    plt.show()
       

def main3():
    def simulate_vel(v_des, dt, N, N1, N2, alpha=alpha):
        vels = np.zeros(N)
        for i in range(N1, N2):
            vels[i] += alpha*dt*(v_des - vels[i-1]) 
        return vels
        
    N = 2000
    dt = 0.004
    timesteps = np.arange(0.0, dt*N,dt)
    v_des = 0.2
    N1 = 100
    N2 = 600
    inputs = np.zeros((N, 7))
    inputs[N1:N2,0] = 3.0 * np.sin(np.arange(N2-N1)*dt *4)
    # inputs[N1:N2,0] = v_des
    # inputs[N1:N2,3] = v_des

    
    r_arm = ApolloInterface(r_arm=True)
    r_arm.go_to_home_position([0.0, 0.0, -np.pi/4, np.pi/2, np.pi/2, np.pi/2, 0.0], 4000, False)
    # r_arm.go_to_posture_array([np.pi/4, 0.0, np.pi/4, np.pi/4, 3*np.pi/4, 3*np.pi/4, 0.0], 2000, False)
    
    poses, velocities, acc, _, u_vec = r_arm.apollo_run_one_iteration(dt, T=dt*len(timesteps), u=inputs, repetitions=1)
    
    plt.figure()
    plt.plot(velocities.squeeze()[:, 0], 'b', label="Measured Velocities")
    plt.plot(inputs[:, 0], 'r', label="Desired Velocities")
    # for a in [2.0,  8.0, 15.0]:
    #     plt.plot(simulate_vel(v_des, dt, N, N1, N2, a), '-', label="Simulated Velocities, alpha="+str(a))
    plt.legend()
    plt.show()
    print()

if __name__ == "__main__":
    main3()