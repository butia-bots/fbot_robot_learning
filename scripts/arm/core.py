import numpy as np

class BaseDorisArm:
    def set_arm_pose(self, pose: np.ndarray):
        raise NotImplementedError()
    
    def get_arm_pose(self)->np.ndarray:
        raise NotImplementedError()