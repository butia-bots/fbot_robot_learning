import numpy as np

class BaseDorisCamera:
    def get_camera_img(self)->np.ndarray:
        raise NotImplementedError