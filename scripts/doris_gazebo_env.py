from doris_env import DorisEnv
from arm.moveit_arm import MoveItArm
from camera.rgb_camera import RGBCamera
from camera.depth_camera import DepthCamera
from gripper.moveit_gripper import MoveItGripper
from mobile_base.navstack_mobile_base import NavStackMobileBase
from neck.sim_neck import SimNeck
from joystick.ros_joystick import ROSJoystick
import rospy

class DorisGazeboEnv(DorisEnv):
    def __init__(self) -> None:
        super(DorisGazeboEnv, self).__init__()
        self.arm = MoveItArm()
        self.rgb_camera = RGBCamera()
        self.depth_camera = DepthCamera()
        self.gripper = MoveItGripper()
        self.mobile_base = NavStackMobileBase()
        self.neck = SimNeck()
        self.joystick = ROSJoystick()

if __name__ == '__main__':
    rospy.init_node('doris_gazebo_env', anonymous=True)
    env = DorisGazeboEnv()
    teleop_agent = env.teleop()
    while not rospy.is_shutdown():
        env.step(teleop_agent.act())