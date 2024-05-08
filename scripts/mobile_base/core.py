import numpy as np

class BaseDorisMobileBase:
    def set_mobile_base_pose(self, pose: np.ndarray):
        raise NotImplementedError()
    
    def get_mobile_base_pose(self)->np.ndarray:
        raise NotImplementedError()