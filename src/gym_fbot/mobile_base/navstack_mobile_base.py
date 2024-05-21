from numpy import ndarray
from gym_fbot.mobile_base.core import BaseDorisMobileBase
from actionlib.simple_action_client import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf import TransformListener
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rospy
import numpy as np

class NavStackMobileBase(BaseDorisMobileBase):
    def __init__(self) -> None:
        super(NavStackMobileBase, self).__init__()
        self.actionlib_client = SimpleActionClient('move_base', MoveBaseAction)
        self.tfl = TransformListener()
        self.cmd_vel = rospy.Publisher('/RosAria/cmd_vel', Twist)
        self.odom = rospy.Subscriber('/RosAria/pose', Odometry, self._update_odom_msg)
        rospy.wait_for_message('/RosAria/pose', Odometry)
    
    def _update_odom_msg(self, msg: Odometry):
        self.odom_msg = msg

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
        self.tfl.waitForTransform('base_footprint', 'map', rospy.Time(), rospy.Duration(10.0))
        translation, rotation  = self.tfl.lookupTransform('base_footprint', 'map', rospy.Time())
        rpy = euler_from_quaternion(rotation)
        return np.array([
            translation[0],
            translation[1],
            rpy[2]
        ])

    def set_mobile_base_vel(self, vel: ndarray):
        twist = Twist()
        twist.linear.x = vel[0]
        twist.angular.z = vel[1]
        self.cmd_vel.publish(twist)

    def get_mobile_base_vel(self)->ndarray:
        return np.array([
            self.odom_msg.twist.twist.linear.x,
            self.odom_msg.twist.twist.angular.z
        ])