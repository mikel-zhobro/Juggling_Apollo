import matplotlib.pyplot as plt
import numpy as np
from apollo_interface.Apollo_It import ApolloInterface

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
       

if __name__ == "__main__":
    main2()