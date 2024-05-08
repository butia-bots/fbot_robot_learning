import numpy as np

class BaseDorisGripper:
    def set_gripper_opening(self, opening: float):
        raise NotImplementedError()
    
    def get_gripper_opening(self)->float:
        raise NotImplementedError()