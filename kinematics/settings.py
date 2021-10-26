import numpy as np

# FILENAME = "/home/apollo/Software/workspace/src/catkin/robots/apollo_robot_model/target.urdf"
FILENAME = "/home/apollo/Software/workspace/src/catkin/playful/playful-kinematics/urdf/apollo.urdf"


WORLD = "universe"
BASE = "BASE"
TCP = "R_PALM"

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


home_pose = np.array([np.pi/8, -0.1, 0.0, 3*np.pi/8, np.pi/2, 0.0, -np.pi/2])  # real model

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

