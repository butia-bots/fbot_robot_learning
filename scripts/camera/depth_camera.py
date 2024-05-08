from camera.core import BaseDorisCamera
import rospy
from sensor_msgs.msg import Image
import ros_numpy
import cv2
import numpy as np

class DepthCamera(BaseDorisCamera):
    def __init__(self) -> None:
        super(DepthCamera, self).__init__()
        self.camera_sub = rospy.Subscriber('/butia_vision/bvb/image_depth', Image, self._update_image_msg)

    def _update_image_msg(self, msg: Image):
        self.image_msg = msg

    def get_camera_img(self) -> np.ndarray:
        image_arr = ros_numpy.numpify(self.image_msg)
        image_arr = cv2.resize(image_arr, (640, 480))
        return image_arr