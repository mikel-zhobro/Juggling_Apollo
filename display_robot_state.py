import matplotlib.pyplot as plt
import numpy as np
from apollo_interface.Apollo_It import ApolloInterface
np.set_printoptions(precision=6, suppress=True)

# home_pose = np.array([1.0, 1.0, np.pi/6, 1.0, np.pi/4, 1.0, 2.0]).reshape(-1,1)
home_pose = np.array([1.0, -1.0, np.pi/6, 1.0, np.pi/4, 1.0, 2.0]).reshape(-1,1)
home_pose = np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.0, np.pi/2]).reshape(-1,1)
# home_pose = np.array([0.0, -0.1, 0.0, np.pi/2, 0.0, 0.0, 0.0]).reshape(-1,1)*0.0


def main():    
    r_arm = ApolloInterface(r_arm=True)
    r_arm.go_to_posture_array(home_pose, 5000, False)
    
    
    p, R = r_arm.get_TCP_pose()
    
    print("R = ")
    print(R)
    print("p = " + str(p.T))

    
if __name__ == "__main__":
    main()