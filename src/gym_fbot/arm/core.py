import numpy as np

class BaseDorisArm:
    def set_arm_pose(self, pose: np.ndarray):
        raise NotImplementedError()
    
    def get_arm_pose(self)->np.ndarray:
        raise NotImplementedError()

    def set_arm_joints(self, joints: np.ndarray):
        raise NotImplementedError()

    def get_arm_joints(self)->np.ndarray:
        raise NotImplementedError()

    def compute_ik(self, pose: np.ndarray)->np.ndarray:
        raise NotImplementedError()