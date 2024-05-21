from numpy import ndarray
from gym_fbot.neck.core import BaseDorisNeck
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import numpy as np

class SimNeck(BaseDorisNeck):
    def __init__(self) -> None:
        super(SimNeck, self).__init__()
        self.horizontal_neck_pub = rospy.Publisher('/doris_head/head_pan_position_controller/command', Float64)
        self.vertical_neck_pub = rospy.Publisher('/doris_head/head_tilt_position_controller/command', Float64)
        self.joint_states = rospy.Subscriber('/doris_head/joint_states', JointState, self._update_joint_state_msg)
        rospy.wait_for_message('/doris_head/joint_states', JointState)

    def _update_joint_state_msg(self, msg: JointState):
        self.joint_state_msg = msg

    def set_angles(self, angles: ndarray):
        self.horizontal_neck_pub.publish(angles[0])
        self.vertical_neck_pub.publish(angles[1])

    def get_angles(self) -> ndarray:
        return np.array([
            self.joint_state_msg.position[self.joint_state_msg.name.index('head_pan_joint')],
            self.joint_state_msg.position[self.joint_state_msg.name.index('head_tilt_joint')],
        ])
