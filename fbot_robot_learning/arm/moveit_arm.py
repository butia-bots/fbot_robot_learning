from numpy import ndarray
import numpy as np
from fbot_robot_learning.arm.core import BaseDorisArm
from moveit_commander.move_group import MoveGroupCommander
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math
import rospy
from moveit_msgs.msg import MoveItErrorCodes

class MoveItArm(BaseDorisArm):
    def __init__(self) -> None:
        super(MoveItArm, self).__init__()
        self.move_group = MoveGroupCommander('arm', robot_description='wx200/robot_description', ns='/wx200')
        self.move_group.set_pose_reference_frame('wx200/base_link')
        self.compute_ik_proxy = rospy.ServiceProxy('/wx200/compute_ik', GetPositionIK)
        self.arm_joints = ['waist', 'shoulder', 'elbow', 'wrist_angle', 'wrist_rotate']
        self.trajectory_controller = rospy.Publisher('/wx200/arm_controller/command', JointTrajectory)

    def compute_ik(self, pose: ndarray)->ndarray:
        ps = PoseStamped()
        ps.pose.position.x = pose[0]
        ps.pose.position.y = pose[1]
        ps.pose.position.z = pose[2]
        quat = quaternion_from_euler(pose[3], pose[4], pose[5])
        ps.pose.orientation.x = quat[0]
        ps.pose.orientation.y = quat[1]
        ps.pose.orientation.z = quat[2]
        ps.pose.orientation.w = quat[3]
        ps.header.frame_id = self.move_group.get_pose_reference_frame()
        req = GetPositionIKRequest()
        req.ik_request.avoid_collisions = False
        req.ik_request.group_name = "arm"
        req.ik_request.robot_state = self.move_group.get_current_state()
        req.ik_request.ik_link_name = self.move_group.get_end_effector_link()
        req.ik_request.pose_stamped = ps
        req.ik_request.timeout = rospy.Duration(1/50.0)
        res: GetPositionIKResponse = self.compute_ik_proxy.call(req)
        joint_values = []
        for i in range(len(res.solution.joint_state.name)):
            if res.solution.joint_state.name[i] in self.arm_joints:
                joint_values.append(res.solution.joint_state.position[i])
        if not (len(joint_values) == 5 or len(joint_values) == 6):
            joint_values = self.get_arm_joints()
        return np.array(joint_values)

    def set_arm_pose(self, pose: ndarray):
        pose = pose.copy()
        pose[5] = math.atan2(pose[1], pose[0])
        joints = self.compute_ik(pose)
        self.set_arm_joints(joints)

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

    def set_arm_joints(self, joints: ndarray):
        #self.move_group.stop()
        #print(joints)
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joints
        traj_point = JointTrajectoryPoint()
        traj_point.positions = joints.tolist()
        traj_point.effort = [20.0,]*len(traj_point.positions)
        #traj_point.velocities = [0.125*np.pi,]*len(traj_point.positions)
        #traj_point.accelerations = [0.25*np.pi,]*len(traj_point.positions)
        traj_point.time_from_start = rospy.Duration(1/50.0)
        trajectory.points = [traj_point,]
        self.trajectory_controller.publish(trajectory)

    def get_arm_joints(self)->ndarray:
        return np.array(self.move_group.get_current_joint_values())