#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Apollo_It.py
@Time    :   2022/02/14
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   defines the interface to the real robot (adapted for ILC)
'''

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
    "R_HR": (-1.9, 3.14 - np.pi/6), #4.0),
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
JOINTS_V_LIMITS = {
    "R_SFE":(-1.91,1.91),
    "R_SAA":(-1.91,1.91),
    "R_HR": (-2.23,2.23),
    "R_EB": (-2.23,2.23),
    "R_WR": (-3.56,3.56),
    "R_WFE":(-3.21,3.21),
    "R_WAA":(-3.21,3.21),

    "L_SFE":(-1.91,1.91),
    "L_SAA":(-1.91,1.91),
    "L_HR": (-2.23,2.23),
    "L_EB": (-2.23,2.23),
    "L_WR": (-3.56,3.56),
    "L_WFE":(-3.21,3.21),
    "L_WAA":(-3.21,3.21)
}


#  Joints |           Joint limits          |  Velocity limits   | Moment limits
# ----------------------------------------------------------------------------------
# A1 (J1) |     +/-170   (-169.59, 169.59)  |   110,0 degree/s   |     176 Nm
# A2 (J2) |     +/-120   (-177.61, -5.72)   |   110,0 degree/s   |     176 Nm
# E1 (J3) |     +/-170   (-108.86, 229.18)  |   128,0 degree/s   |     100 Nm
# A3 (J4) |     +/-120   (-119.74, 119.74)  |   128,0 degree/s   |     100 Nm
# A4 (J5) |     +/-170   (-177.61, 77.34)   |   204,0 degree/s   |     100 Nm
# A5 (J6) |     +/-120   (-119.74, 119.74)  |   184,0 degree/s   |     38 Nm
# A6 (J7) |     +/-170   (-169.59, 169.59)  |   184,0 degree/s   |     38 Nm


# O8o helper functions
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
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
        apollo.add_position_command(joint, float(position), target_iteration, True)

    # sending stack to robot and waiting for the desired number of iterations
    if not bursting:  # if blocking:
        observation = apollo.iteration_sync(target_iteration)  # waits until the queue of commmands has been executed
    else:
        observation = apollo.burst(nb_iterations)
    return observation


def read(nb_iterations=None):
    """Returns an observation
    """
    observation=None
    if nb_iterations:
        # SL current iteration
        observation = apollo.pulse()
        current_iteration = observation.get_iteration()
        target_iteration = current_iteration + nb_iterations
        observation = apollo.iteration_sync(target_iteration)
    else:
        observation = apollo.read()
    return observation
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------


class ApolloInterface:
    def __init__(self, r_arm=True):
        """ An interface for Apollo that simplifies the ILC related operations.
            More precisely, it implements feed forward control for whole trajectories
            (see apollo_run_one_iteration and apollo_run_one_iteration2 for the 1DOF control loop
             and apollo_run_one_iteration_with_feedback for the 2DOF control loop )

        Args:
            r_arm (bool, optional): _description_. Defaults to True.
        """
        self.r_arm = r_arm
        if r_arm:
            self.joints_list = R_joints
        else:
            self.joints_list = L_joints

    def obs_to_numpy(self, obs, des=False):
        """Transform a O8O observation to a numpy matrix

        Args:
            obs ([type]): O8O observation

        Returns:
            [7, 4] numpy array:  angle, angle_velocity, angle_acceleration, sensed_torque
        """
        # first for loop traverses the joints, second one the angle, ang_vel, ang_acc of the end effector!!!
        obs_o = obs.get_observed_states()

        obs_np = np.zeros((len(self.joints_list), 4))
        for n, joint in enumerate(self.joints_list):
            i = jointsToIndexDict[joint]
            for k in range(3):
                obs_np[n][k] = obs_o.get(i).get()[k]
            obs_np[n][3] = obs_o.get(i).get_sensed_load()

        if not des:
            return obs_np

        obs_o = obs.get_desired_states()
        des_np = np.zeros((len(self.joints_list), 4))
        for n, joint in enumerate(self.joints_list):
            i = jointsToIndexDict[joint]
            for k in range(3):
                des_np[n][k] = obs_o.get(i).get()[k]
            des_np[n][3] = obs_o.get(i).get_sensed_load()

        return obs_np, des_np

    def read(self, nb_iteration=None, des=False):
        """
        Args:
            des (bool): whether to return the desired states
            nb_iteration (int): If not None, waits nb_iteration before reading.

        Returns:
            obs or (obs, des)
        """
        return self.obs_to_numpy(read(nb_iteration), des=des)

    def apollo_run_one_iteration2(self, dt, T, u, joint_home_config=None, repetitions=1, it=0, go2position=False):
        """ Runs the system for the time interval 0->T

        Args:
            dt ([double]): timestep
            T ([double]): approximate time required for the iteration
            joint_home_config ([list]): home_state(if set the robot has to first go there before runing the inputs)
            u ([np.array(double)]): inputs of the shape [N_steps, 7(n_joints)]
            repetitions (int, optional): Nr of times to repeat the given input trajectory.

        Returns:
            [np.array(R, N, 7, 1)]: x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, u_vec of shape [R, N, 7, 1]
        """
        N0 = len(u)
        assert N0>0, "Please give a valid feed forward input"
        assert abs(T-N0*dt) <= dt, "Input signal is of length {} instead of length {}".format(len(u)*dt ,T)

        real_homes = np.zeros((repetitions, 7,1))
        if joint_home_config is not None:
            self.go_to_home_position(joint_home_config)[:,0:1] # we dont directly save this, but real_homes[r] = thetas_s[r,0], does that implicitely

        # Vectors to collect the history of the system states
        N = N0 * repetitions + 1
        n_joints = 7
        thetas_s = np.zeros((repetitions, N0, n_joints, 1))
        vel_s    = np.zeros((repetitions, N0, n_joints, 1))
        acc_s    = np.zeros((repetitions, N0, n_joints, 1))
        dP_N_vec = np.zeros((repetitions, N0, n_joints, 1))

        # Action Loop
        delta_it = int(1000*dt)
        for r in range(repetitions):
            for i in range(N0):
                # one step simulation
                if go2position:
                    obs_np = self.go_to_posture_array(u[i], delta_it, globs.bursting)
                else:
                    obs_np = self.go_to_speed_array(u[i], delta_it, globs.bursting)
                # collect state of the system
                thetas_s[r,i] = obs_np[:,0].reshape(7, 1)
                vel_s[r,i] = obs_np[:,1].reshape(7, 1)
                acc_s[r,i] = obs_np[:,2].reshape(7, 1)
                # collect helpers
                dP_N_vec[r,i] = obs_np[:,3].reshape(7, 1)
            real_homes[r] = thetas_s[r,0]
        # obs = self.go_to_speed_array(np.zeros((7,1)), 1000, globs.bursting)
        # obs = self.go_to_posture_array(thetas_s[-1,-1], 1000, globs.bursting)
        return thetas_s, vel_s, acc_s, dP_N_vec, u, real_homes

    def apollo_run_one_iteration_with_feedback(self, dt, T, u, thetas_des, P=0.07, joint_home_config=None, repetitions=1, it=0, go2position=False):
        """ Runs the system for the time interval 0->T

        Args:
            dt ([double]): timestep
            T ([double]): approximate time required for the iteration
            joint_home_config ([list]): home_state(if set the robot has to first go there before runing the inputs)
            u ([np.array(double)]): [N_steps, 7(n_joints)] inputs of the shape
            thetas_des ([np.array(double)]): [N_steps, 7(n_joints)] desired joint trajectory with which we perform feedback control
            repetitions (int, optional): Nr of times to repeat the given input trajectory.

        Returns:
            [np.array(R, N, 7, 1)]: x_b, u_b, x_p, u_p, dP_N_vec, gN_vec, u_vec of shape [R, N, 7, 1]
        """
        N0 = len(u)
        assert N0>0, "Please give a valid feed forward input"
        assert abs(T-N0*dt) <= dt, "Input signal is of length {} instead of length {}".format(len(u)*dt ,T)

        real_homes = np.zeros((repetitions, 7,1))
        if joint_home_config is not None:
            real_homes[0] = self.go_to_home_position(joint_home_config)[:,0:1]

        # Vectors to collect the history of the system states
        N = N0 * repetitions + 1
        n_joints = 7
        thetas_s = np.zeros((repetitions, N0, n_joints, 1))
        vel_s    = np.zeros((repetitions, N0, n_joints, 1))
        acc_s    = np.zeros((repetitions, N0, n_joints, 1))
        dP_N_vec = np.zeros((repetitions, N0, n_joints, 1))

        # Action Loop
        delta_it = int(1000*dt)
        feedback = np.array(P).reshape(-1, 1)*(thetas_des[1] - real_homes[0])
        for r in range(repetitions):
            for i in range(N0):
                # one step simulation
                if go2position:
                    obs_np = self.go_to_posture_array(u[i], delta_it, globs.bursting)
                else:
                    obs_np = self.go_to_speed_array(u[i]+feedback, delta_it, globs.bursting)
                # collect state of the system
                thetas_s[r,i] = obs_np[:,0].reshape(7, 1)
                vel_s[r,i] = obs_np[:,1].reshape(7, 1)
                acc_s[r,i] = obs_np[:,2].reshape(7, 1)
                # collect helpers
                dP_N_vec[r,i] = obs_np[:,3].reshape(7, 1)
                # calculate feedback
                feedback = np.array(P).reshape(-1, 1)*(thetas_des[(i+1)%N0] - thetas_s[r,i])
            if r == repetitions-1:
                break
            real_homes[r] = thetas_s[r,-1]
        return thetas_s, vel_s, acc_s, dP_N_vec, u, real_homes

    def measure_extras(self, dt, time):
        """ A simple function that takes measurements for a certain time interval without sending any input.

        Args:
            dt (float): timestep to set the control frequency
            time (float): time interval we want to measure for

        Returns:
            thetas_s, vel_s (np.array(N, 7, 1)):        the actual position, velocity trajecytories during this time
            thetas_dess, vel_dess (np.array(N, 7, 1)):  the desired position, velocity that Apollo sees during this time
        """
        N_extra = int(time/dt)
        thetas_s = np.zeros((N_extra, 7, 1))
        vel_s    = np.zeros((N_extra, 7, 1))
        thetas_dess    = np.zeros((N_extra, 7, 1))
        vel_dess = np.zeros((N_extra, 7, 1))
        delta_it = int(1000*dt)
        for i in range(N_extra):
            obs_np, des_np = self.read(delta_it, des=True)
            thetas_s[i]         = obs_np[:, 0].reshape(7,1)
            vel_s[i]            = obs_np[:, 1].reshape(7,1)
            thetas_dess[i]      = des_np[:, 0].reshape(7,1)
            vel_dess[i]         = des_np[:, 1].reshape(7,1)
        return thetas_s, vel_s, thetas_dess, vel_dess

    def go_to_speed_array(self, speeds, nb_iterations, bursting, override=True):
        """Move right arm to a certain joint configuration.

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
        """Move right arm to a certain joint configuration.

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

    def go_to_home_position(self, home_pose=None, it_time=4000, eps=3., wait=2000, zero_speed=True, verbose=True, reset_PID=True):
        """Method that brings the arm to the home position.
           It uses inverse dynamics to compute feed forward control to go close to the desired position.
           Finally it uses a PID feedback controller to improve the accuracy.

        Args:
            home_pose (np.array(7,1)):  Joint configuration describing the home position.
            it_time (int):              nr of iteration the inverse dynamics should optimize for
            eps (float):                accuracy described in degree
            wait (int):                 nr of iterations the arm should stay in the desired configuration before calling it a day
            zero_speed (bool):          If set we sent zero desired joint velocities after achieving the goal configuration.
            verbose (bool):             Whether to print verbose information
            reset_PID (bool):           Whether to reset the PID controller in the beginning

        Returns:
            _type_: _description_
        """
        # eps: degree
        if home_pose is None:
            home_pose = np.zeros((7,1))
            home_pose[1] = -0.2

        i = 0 if reset_PID else 1
        dt = 0.002 if reset_PID else 0.01
        P = 1.6
        I = 0.04
        D = 0.2
        if reset_PID:
            self.error_P = np.zeros(7)
            self.error_I = np.zeros(7)
            self.error_D = np.zeros(7)
        if False:
            print("Using IK_dynamics")
            while True:
                obs = self.go_to_posture_array(home_pose, it_time, bursting=globs.bursting, override=True)
                print("HOME with error:", np.linalg.norm(np.array(home_pose).squeeze()-obs[:,0].squeeze()))
                if np.linalg.norm(np.array(home_pose).squeeze()-obs[:,0].squeeze()) <= eps:
                    break
        else:
            if verbose:
                print("Using PID control at the end")

            obs = self.read()
            error = np.array(home_pose).squeeze()-obs[:,0].squeeze()

            # for i in range(int(5./dt)): # try for 5 sec
            i = 0; k = 0
            while True and k < wait:
                if all((180.0/np.pi)*np.abs(error) <= eps):
                    k +=1
                if i==0:
                    obs = self.go_to_posture_array(home_pose, it_time, globs.bursting)
                    print("HOME ERROR", np.array(home_pose).squeeze()-obs[:,0].squeeze())
                i += 1
                error = np.array(home_pose).squeeze()-obs[:,0].squeeze()
                self.error_D = (error - self.error_P)/dt
                self.error_P = error
                self.error_I += self.error_P*dt

                controller_Input = self.error_P*P + self.error_I*I + self.error_D*D
                obs = self.go_to_speed_array(controller_Input, int(dt*1000), globs.bursting)
        if zero_speed:
            obs = self.go_to_speed_array(np.zeros_like(home_pose), it_time/4, globs.bursting)
        if verbose:
            print("{}. HOME with error: {} deg".format(i, (180.0 / np.pi)*np.linalg.norm(np.array(home_pose).squeeze()-obs[:,0].squeeze())))
            print((180.0 / np.pi)*(np.array(home_pose).squeeze()- obs[:,0].squeeze()))
        return obs[:,0:2].reshape(7, 2)

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
    r_arm.go_to_home_position(verbose=True)


def main2():
    for j in joints[:7]:
        p = JOINTS_LIMITS[j]
        A = 180./np.pi
        print ((p[0] *A, p[1]*A))
if __name__ == "__main__":
    main()