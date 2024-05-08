import numpy as np

class BaseDorisNeck:
    def set_angles(self, angles: np.ndarray):
        raise NotImplementedError()
    
    def get_angles(self)->np.ndarray:
        raise NotImplementedError()