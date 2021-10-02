import numpy as np
import O8O_apollo as apollo
import matplotlib.pyplot as plt


# List of all joints(motors) for Apollo. 
# Their position in the list is the index to communcate through. 
joints = [
 "R_SFE", "R_SAA", "R_HR", "R_EB", "R_WR", "R_WFE", "R_WAA",  # Right Arm
 "L_SFE", "L_SAA", "L_HR", "L_EB", "L_WR", "L_WFE", "L_WAA",  # Left Arm
 "B_HN", "B_HT", "B_HR",   "R_EP", "R_ET", "L_EP", "L_ET",    # Head
 "R_FR", "R_RF", "R_MF",   "R_LF",                            # Right Hand
 "L_FR", "L_RF", "L_MF",   "L_LF"                             # Left Hand
]

R_joints = ["R_SFE", "R_SAA", "R_HR", "R_EB", "R_WR", "R_WFE", "R_WAA"]
L_joints = ["L_SFE", "L_SAA", "L_HR", "L_EB", "L_WR", "L_WFE", "L_WAA"]

jointsToIndexDict = {
 "R_SFE": 0, "R_SAA": 1, "R_HR": 2,    "R_EB": 3,  "R_WR": 4,  "R_WFE": 5,  "R_WAA": 6,  # Right Arm
 "L_SFE": 7, "L_SAA": 8, "L_HR": 9,    "L_EB": 10, "L_WR": 11, "L_WFE": 12, "L_WAA": 13, # Left Arm
 "B_HN": 14, "B_HT": 15, "B_HR": 16,   "R_EP": 17, "R_ET": 18, "L_EP": 19,  "L_ET": 20,  # Head
 "R_FR": 21, "R_RF": 22, "R_MF": 23,   "R_LF": 24,                                       # Right Hand
 "L_FR": 25, "L_RF": 26, "L_MF": 27,   "L_LF": 28                                        # Left Hand
}

jointsToUse = ["R_SFE", "R_SAA", "R_HR", "R_EB", "R_WR", "R_WFE", "R_WAA"]


def ref_name_to_index(posture):
    """Transforms {joint_name: joint_posture} to {joint_index: joint_posture}

    Args:
        posture ([dict]): {joint_name: joint_posture}

    Returns:
        [dict]: {joint_index: joint_posture}
    """
    return { jointsToIndexDict[joint]: posture for joint, posture in posture.items()}


def go_to_speed(speeds, nb_iterations, bursting):
    """Move robot to a certain speed-joint configuration.

    Args:
        speeds ([dict]): the joint configuration
        nb_iterations ([int]): number of iteration this movement should take(in how many tics)
        bursting ([bool]):
    """
    # implementation by Vberenz
    # SL current iteration
    observation = apollo.pulse()
    current_iteration = observation.get_iteration()
    target_iteration = current_iteration + nb_iterations

    # creating command stacks
    for joint, speed in speeds.iteritems():
        apollo.add_speed_command(joint, speed, target_iteration, True)

    # sending stack to robot and waiting for the desired number of iterations
    if not bursting:  # if blocking:
        observation = apollo.iteration_sync(target_iteration)  # waits until the queue of commmands has been executed
    else:
        observation = apollo.burst(nb_iterations)
    return observation


def go_to_posture(posture, nb_iterations, bursting):
    """Move robot to a certain joint configuration.

    Args:
        posture ([dict]): the joint configuration
        nb_iterations ([int]): number of iteration this movement should take(in how many tics)
        bursting ([bool]):
    """
    # SL current iteration
    observation = apollo.pulse()
    current_iteration = observation.get_iteration()
    target_iteration = current_iteration + nb_iterations

    # creating command stacks
    for joint, position in posture.iteritems():
        apollo.add_position_command(joint, position, target_iteration, False)

    # sending stack to robot and waiting for the desired number of iterations
    if not bursting:  # if blocking:
        observation = apollo.iteration_sync(target_iteration)  # waits until the queue of commmands has been executed
    else:
        observation = apollo.burst(nb_iterations)
    return observation


class MyApollo:
    def __init__(self, r_arm=True):
        if r_arm:
            self.joints_list = R_joints
        else:
            self.joints_list = L_joints
            
    def obs_to_numpy(self, obs):
        """Transform a O8O observation to a numpy matrix

        Args:
            obs ([type]): O8O observation

        Returns:
            [type]: a [10, 3] numpy array
        """
        # first for loop traverses the joints, second one the angle, ang_vel, ang_acc of the end effector!!!
        obs = obs.get_observed_states()
        
        obs_np = np.zeros((len(self.joints_list), 3))
        for joint in self.joints_list:
            i = jointsToIndexDict[joint]
            for k in range(3):
                obs_np[i][k] = obs.get(i).get()[k]
        return obs_np

    def apollo_run_one_iteration(self, dt, T, u, x0=None, repetitions=1, it=0):
        """ Runs the system for the time interval 0->T

        Args:
            dt ([double]): timestep
            T ([double]): approximate time required for the iteration
            x0 ([list]): home_state(if set the robot has to first go there before runing the inputs)
            u ([np.array(double)]): inputs of the shape [N_steps, 7(n_joints)]
            repetitions (int, optional): Nr of times to repeat the given input trajectory.

        Returns:
            [type]: x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, u_vec of shape [1, N]
        """
        assert abs(T-len(u)*dt) <= dt, "Input signal is of length {} instead of length {}".format(len(u)*dt ,T)

        N0 = len(u)
        u = np.squeeze(np.tile(u, [repetitions, 1]))

        if x0 is not None:
            self.go_to_speed_array([0.0, 0.0], 2000, False)

        # Vectors to collect the history of the system states
        N = N0 * repetitions + 1
        n_joints = 7
        x_s = np.zeros((N, n_joints));  # x_b[0] = x0
        u_s = np.zeros((N,n_joints))
        a_s = np.zeros((N,n_joints))
        dP_N_vec = np.zeros_like(x_s)  # TODO: hand torque sensor

        # Action Loop
        # dt = 0.004
        delta_it = int(1000*dt)
        for i in range(N-1):
            # one step simulation
            obs_np = self.go_to_speed_array(u[i], delta_it, False)
            # collect state of the system
            x_s[i+1] = obs_np[:,0]
            u_s[i+1] = obs_np[:,1]
            a_s[i+1] = obs_np[:,2]
            # collect helpers
            dP_N_vec[i+1] = 0
        if x0 is not None:
            return x_s, u_s, a_s, dP_N_vec
        else:
            return x_s[1:], u_s[1:], a_s[1:],dP_N_vec[1:]

    def go_to_speed_array(self, speeds, nb_iterations, bursting):
        """Move right arm to a certain joint configuration and reset pinocchio jointstates.

        Args:
            speeds ([array]): [7x1] the joint configuration as array
            nb_iterations ([int]): number of iteration this movement should take(in how many tics)
            bursting ([bool]):
        """
        speeds_dict = {self.joints_list[i]: p for i, p in enumerate(speeds)}
        speeds_dict = ref_name_to_index(speeds_dict)
        observation = go_to_speed(speeds_dict, nb_iterations, bursting)
        obs_np = self.obs_to_numpy(observation)
        return obs_np

    def go_to_posture_array(self, posture, nb_iterations, bursting):
        """Move right arm to a certain joint configuration and reset pinocchio jointstates.

        Args:
            posture ([array]): [7x1] the joint configuration as array
            nb_iterations ([int]): number of iteration this movement should take(in how many tics)
            bursting ([bool]):
        """
        posture_dict = {self.joints_list[i]: p for i, p in enumerate(posture)}
        posture_dict = ref_name_to_index(posture_dict)
        observation = go_to_posture(posture_dict, nb_iterations, bursting)
        obs_np = self.obs_to_numpy(observation)
        return obs_np

def main():
    # go_to_posture_array([np.pi/4, 0.0, np.pi/4, np.pi/4, 3*np.pi/4, 3*np.pi/4, 0.0], 2000, False)

    N = 2000
    dt = 0.004
    timesteps = np.arange(0.0, dt*N,dt)
    inputs = np.zeros((N, 7))
    inputs[:,0] = 0.3 * np.sin(timesteps)
    
    r_arm = MyApollo(r_arm=True)
    r_arm.go_to_posture_array([0.0, 0.0, -np.pi/4, np.pi/2, np.pi/2, np.pi/2, 0.0], 2000, False)
    
    poses, velocities, acc, _ = r_arm.apollo_run_one_iteration(dt, T=dt*len(timesteps), u=inputs)
    
    plt.figure()
    plt.plot(timesteps, poses[:, 0], label='angle')
    plt.plot(timesteps, velocities[:, 0], label='velocity')
    plt.plot(timesteps, inputs[:, 0], label='des_velocity')
    plt.plot(timesteps, acc[:, 0], label='acc')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()