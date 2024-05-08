from numpy import ndarray
import numpy as np
from arm.core import BaseDorisArm
from moveit_commander.move_group import MoveGroupCommander
from tf.transformations import euler_from_quaternion
import math

class MoveItArm(BaseDorisArm):
    def __init__(self) -> None:
        super(MoveItArm, self).__init__()
        self.move_group = MoveGroupCommander('arm', robot_description='doris_arm/robot_description', ns='/doris_arm')
        self.move_group.set_pose_reference_frame('doris_arm/base_link')

    def set_arm_pose(self, pose: ndarray):
        pose = pose.copy()
        pose[5] = math.atan2(pose[1], pose[0])
        self.move_group.stop()
        self.move_group.set_pose_target(pose.tolist())
        self.move_group.go(wait=False)

    def get_arm_pose(self) -> ndarray:
        ps = self.move_group.get_current_pose()
        position = [
            ps.pose.position.x,
            ps.pose.position.y,
            ps.pose.position.z
        ]
        rpy = euler_from_quaternion([
            ps.pose.orientation.x,
            ps.pose.orientation.y,
            ps.pose.orientation.z,
            ps.pose.orientation.w
        ])
        return np.concatenate([position, rpy])