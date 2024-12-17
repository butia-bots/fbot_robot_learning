from gym_fbot.gripper.core import BaseDorisGripper
from moveit_commander.move_group import MoveGroupCommander
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import rospy

class MoveItGripper(BaseDorisGripper):
    def __init__(self) -> None:
        super(MoveItGripper, self).__init__()
        self.move_group = MoveGroupCommander('gripper', robot_description='wx200/robot_description', ns='/wx200')
        self.arm_joints = ['left_finger', 'right_finger']
        self.trajectory_controller = rospy.Publisher('/wx200/gripper_controller/command', JointTrajectory)

    def set_gripper_joints(self, joints: np.ndarray):
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

    def set_gripper_opening(self, opening: float):
        #self.move_group.stop()
        opening = np.clip(opening, 0.0, 0.1)
        #opening = 0.1
        self.set_gripper_joints(np.array([opening/2.0,-opening/2.0]))
    
    def get_gripper_opening(self) -> float:
        return sum([abs(j) for j in self.move_group.get_current_joint_values()])
