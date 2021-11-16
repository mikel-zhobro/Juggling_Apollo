import numpy as np
import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation

import globs
try:
    import O8O_apollo as apollo
except:
    pass


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

home_pose = np.array([np.pi/8, 0.0, 0.0, 3*np.pi/8, np.pi/2, 0.0, -np.pi/2])  # real model

JOINTS_LIMITS = {
    "R_SFE":(-2.96,2.96),
    "R_SAA":(-3.1,-0.1),
    "R_HR": (-1.9,4.0),
    "R_EB": (-2.09,2.09),
    "R_WR": (-3.1,1.35),
    "R_WFE":(-2.09,2.09),
    "R_WAA":(-2.96,2.96),

    "L_SFE":(-2.96,2.96),
    "L_SAA":(-3.1,-0.1),
    "L_HR": (-1.9,4.0),
    "L_EB": (-2.09,2.09),
    "L_WR": (-3.1,1.35),
    "L_WFE":(-2.09,2.09),
    "L_WAA":(-2.96,2.96)
}


def ref_name_to_index(posture):
    """Transforms {joint_name: joint_posture} to {joint_index: joint_posture}

    Args:
        posture ([dict]): {joint_name: joint_posture}

    Returns:
        [dict]: {joint_index: joint_posture}
    """
    return { jointsToIndexDict[joint]: posture for joint, posture in posture.items()}


def go_to_speed(speeds, nb_iterations, bursting, override=False):
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
        apollo.add_speed_command(joint, float(speed), target_iteration, True)

    # sending stack to robot and waiting for the desired number of iterations
    if not bursting:  # if blocking:
        observation = apollo.iteration_sync(target_iteration)  # waits until the queue of commmands has been executed
    else:
        observation = apollo.burst(nb_iterations)
    return observation


def go_to_posture(posture, nb_iterations, bursting, override=False):
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
        apollo.add_position_command(joint, float(position), target_iteration, override)

    # sending stack to robot and waiting for the desired number of iterations
    if not bursting:  # if blocking:
        observation = apollo.iteration_sync(target_iteration)  # waits until the queue of commmands has been executed
    else:
        observation = apollo.burst(nb_iterations)
    return observation


class ApolloInterface:
    def __init__(self, r_arm=True):
        self.r_arm = r_arm
        if r_arm:
            self.joints_list = R_joints
        else:
            self.joints_list = L_joints

    def obs_to_numpy(self, obs):
        """Transform a O8O observation to a numpy matrix

        Args:
            obs ([type]): O8O observation

        Returns:
            [type]: a [10, 4] numpy array:  angle, angle_velocity, angle_acceleration, sensed_torque
        """
        # first for loop traverses the joints, second one the angle, ang_vel, ang_acc of the end effector!!!
        obs = obs.get_observed_states()

        obs_np = np.zeros((len(self.joints_list), 4))
        for joint in self.joints_list:
            i = jointsToIndexDict[joint]
            for k in range(3):
                obs_np[i][k] = obs.get(i).get()[k]
            obs_np[i][3] = obs.get(i).get_sensed_load()
        return obs_np

    def apollo_run_one_iteration(self, dt, T, u, joint_home_config=None, repetitions=1, it=0, go2position=False):
        """ Runs the system for the time interval 0->T

        Args:
            dt ([double]): timestep
            T ([double]): approximate time required for the iteration
            joint_home_config ([list]): home_state(if set the robot has to first go there before runing the inputs)
            u ([np.array(double)]): inputs of the shape [N_steps, 7(n_joints)]
            repetitions (int, optional): Nr of times to repeat the given input trajectory.

        Returns:
            [type]: x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, u_vec of shape [1, N]
        """
        assert abs(T-len(u)*dt) <= dt, "Input signal is of length {} instead of length {}".format(len(u)*dt ,T)

        N0 = len(u)
        u = np.squeeze(np.tile(u, [repetitions, 1]))

        if joint_home_config is not None:
            self.go_to_home_position(joint_home_config)

        # Vectors to collect the history of the system states
        N = N0 * repetitions + 1
        n_joints = 7
        thetas_s = np.zeros((N, n_joints, 1));  thetas_s[0] = joint_home_config
        vel_s    = np.zeros((N, n_joints, 1))
        acc_s    = np.zeros((N, n_joints, 1))
        dP_N_vec = np.zeros((N, n_joints, 1))

        # Action Loop
        delta_it = int(1000*dt)
        for i in range(N-1):
            # one step simulation
            if go2position:
                obs_np = self.go_to_posture_array(u[i], delta_it, globs.bursting)
            else:
                obs_np = self.go_to_speed_array(u[i], delta_it, globs.bursting)
            # collect state of the system
            thetas_s[i+1] = obs_np[:,0].reshape(7, 1)
            vel_s[i+1] = obs_np[:,1].reshape(7, 1)
            acc_s[i+1] = obs_np[:,2].reshape(7, 1)
            # collect helpers
            dP_N_vec[i+1] = obs_np[:,3].reshape(7, 1)

        if joint_home_config is not None:
            return thetas_s, vel_s, acc_s, dP_N_vec, u
        else:
            return thetas_s[1:], vel_s[1:], acc_s[1:],dP_N_vec[1:], u

    def go_to_speed_array(self, speeds, nb_iterations, bursting, override=True):
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

    def go_to_posture_array(self, posture, nb_iterations, bursting, override=False):
        """Move right arm to a certain joint configuration and reset pinocchio jointstates.

        Args:
            posture ([array]): [7x1] the joint configuration as array
            nb_iterations ([int]): number of iteration this movement should take(in how many tics)
            bursting ([bool]):
        """
        posture_dict = {self.joints_list[i]: p for i, p in enumerate(posture)}
        posture_dict = ref_name_to_index(posture_dict)
        observation = go_to_posture(posture_dict, nb_iterations, bursting, override)
        obs_np = self.obs_to_numpy(observation)
        return obs_np

    def go_to_home_position(self, home_pose=None, it_time=4000):
        eps = 1e-4 # 2e-3  # <2mm
        # eps = 2e-3
        if home_pose is None:
            home_pose = np.zeros((7,1))

        if False:
            print("Using IK_dynamics")
            while True:
                obs = self.go_to_posture_array(home_pose, it_time, bursting=globs.bursting, override=True)
                print("HOME with error:", np.linalg.norm(np.array(home_pose).squeeze()-obs[:,0].squeeze()))
                if np.linalg.norm(np.array(home_pose).squeeze()-obs[:,0].squeeze()) <= eps:
                    break
        else:
            dt = 0.002
            P = 1.6
            I = 0.04
            D = 0.2
            print("Not using IK_dynamics")
            obs = self.go_to_posture_array(home_pose, it_time, globs.bursting)

            error_P = np.zeros(7)
            error_I = np.zeros(7)
            error_D = np.zeros(7)
            while True:
                error = np.array(home_pose).squeeze()-obs[:,0].squeeze()
                error_D = (error - error_P)/dt
                error_P = error
                error_I += error_P*dt

                controller_Input = error_P*P + error_I*I + error_D*D
                obs = self.go_to_speed_array(controller_Input, int(dt*1000), globs.bursting)
                # print("HOME with error:", np.linalg.norm(error_P))
                # print(error)
                # print(error_P*P)
                # print(error_I*I)
                # print(error_D*D)
                # print(error_D)
                if np.linalg.norm(error_P) <= eps:
                    break

        obs = self.go_to_speed_array(np.zeros_like(home_pose), it_time/4, globs.bursting)
        print("HOME with error: {} mm".format(np.linalg.norm(np.array(home_pose).squeeze()-obs[:,0].squeeze())))
        return obs[:,0].reshape(7, 1)

    # def get_TCP_pose(self):
    #     observation = apollo.read()
    #     cartesian_states = observation.get_cartesian()
    #     if self.r_arm:
    #         hand = cartesian_states.hands[0]
    #     else:
    #         hand = cartesian_states.hands[1]

    #     return np.array(hand.position).reshape(-1,1), Rotation.from_quat(hand.orientation).as_dcm()




def plot_simulation(dt, u, thetas_s, vel_s, acc_s, dP_N_vec=None, thetas_s_des=None, title=None, vertical_lines=None, horizontal_lines=None):
  # Everything are column vectors
  for i in range(thetas_s.shape[1]):
    fig, axs = plt.subplots(4, 1)
    timesteps = np.arange(u.shape[0]) * dt
    axs[0].plot(timesteps, thetas_s[:,i], label='Theta_{} [rad]'.format(i))
    if horizontal_lines is not None:
        for pos, label in horizontal_lines.items():
            axs[0].axhline(pos, linestyle='--', color='brown')  # , label=label
    if thetas_s_des is not None:
        axs[0].plot(timesteps, thetas_s_des, color='green', linestyle='dashed', label='Desired')
    axs[1].plot(timesteps, vel_s[:,i], label='w_{} [rad/s]'.format(i))
    axs[2].plot(timesteps, acc_s[:,i], 'r', label='a_{} [rad/s^2]'.format(i))
    if dP_N_vec is not None:
        axs[2].plot(timesteps, dP_N_vec[:,i], label='dP_N')
    axs[3].plot(timesteps, u[:,i], 'b', label='u_in_{} [rad/s]'.format(i))

    for ax in axs:
        if vertical_lines is not None:
            for pos in vertical_lines:
                ax.axvline(pos, linestyle='--', color='k')
        ax.legend(loc=1)

    if title is not None:
        fig.suptitle(title)
  plt.show(block=True)


def main():
    # go_to_posture_array([np.pi/4, 0.0, np.pi/4, np.pi/4, 3*np.pi/4, 3*np.pi/4, 0.0], 2000, False)
    rep = 2
    N = 2000
    dt = 0.004
    timesteps = np.arange(0.0, dt*N,dt)
    inputs = np.zeros((N, 7))
    inputs[:,1] = 0.3 * np.sin(timesteps)
    inputs[:,3] = 0.3 * np.sin(timesteps)
    inputs[:,4] = 0.3 * np.sin(timesteps)
    inputs[:,6] = 0.3 * np.sin(timesteps)

    r_arm = ApolloInterface(r_arm=True)
    print("GOING HOME!")
    r_arm.go_to_home_position()

    if False:
        # Run apollo
        poses, velocities, acc, _, u = r_arm.apollo_run_one_iteration(dt, T=dt*len(timesteps), u=inputs, repetitions=rep)

        # Test plot_simulation
        # poses = np.random.rand(*inputs.shape)
        # velocities = np.random.rand(*inputs.shape)
        # acc = np.random.rand(*inputs.shape)

        plot_simulation(dt, u, poses, velocities, acc)

if __name__ == "__main__":
    main()