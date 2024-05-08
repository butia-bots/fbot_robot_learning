from numpy import ndarray
from mobile_base.core import BaseDorisMobileBase
from actionlib.simple_action_client import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf import TransformListener
import rospy
import numpy as np

class NavStackMobileBase(BaseDorisMobileBase):
    def __init__(self) -> None:
        super(NavStackMobileBase, self).__init__()
        self.actionlib_client = SimpleActionClient('move_base', MoveBaseAction)
        self.tfl = TransformListener()
    
    def set_mobile_base_pose(self, pose: ndarray):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.pose.position.x = pose[0]
        goal.target_pose.pose.position.y = pose[1]
        quat = quaternion_from_euler(0.0, 0.0, pose[2])
        goal.target_pose.pose.orientation.x = quat[0]
        goal.target_pose.pose.orientation.y = quat[1]
        goal.target_pose.pose.orientation.z = quat[2]
        goal.target_pose.pose.orientation.w = quat[3]
        self.actionlib_client.cancel_all_goals()
        self.actionlib_client.send_goal(goal)

    def get_mobile_base_pose(self) -> ndarray:
        translation, rotation  = self.tfl.lookupTransform('base_footprint', 'map', rospy.Time(), rospy.Duration(10.0))
        rpy = euler_from_quaternion(rotation)
        return np.array([
            translation[0],
            translation[1],
            rpy[2]
        ])
