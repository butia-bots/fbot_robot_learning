from typing import List, Tuple
from gymnasium import Env
from gymnasium.spaces import Box, Dict
import numpy as np
from gym_fbot.arm.core import BaseDorisArm
from gym_fbot.camera.core import BaseDorisCamera
from gym_fbot.gripper.core import BaseDorisGripper
from gym_fbot.mobile_base.core import BaseDorisMobileBase
from gym_fbot.neck.core import BaseDorisNeck
from gym_fbot.joystick.core import BaseJoystick
from collections import namedtuple
import math
import rospy


class DorisEnv(Env):
    def __init__(self) -> None:
        #(arm_0, arm_1, arm_2, arm_3, arm_4, gripper_opening, base_linear_x, base_angular_z, neck_horizontal, neck_vertical)
        self.action_min = np.array([-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,0.0,-1.0,-2*np.pi,-2*np.pi,-2*np.pi], dtype=float)
        self.action_max = np.array([2*np.pi,2*np.pi,2*np.pi,2*np.pi,2*np.pi,0.12,1.0,2*np.pi,2*np.pi,2*np.pi], dtype=float)
        self.proprio_min = self.action_min
        self.proprio_max = self.action_max
        self.action_space = Box(
            low=self.action_min,
            high=self.action_max,
            shape=(10,),
            dtype=float
        )
        self.observation_space = Dict({
            "pixels": Dict({
                "top": Box(
                                low=0,
                                high=255,
                                shape=(480, 640, 3),
                                dtype=np.uint8
                            ),
                "wrist": Box(
                                low=0,
                                high=255,
                                shape=(480, 640, 3),
                                dtype=np.uint8
                            ),
            }),
            "agent_pos": Box(
                low=self.proprio_min,
                high=self.proprio_max,
                shape=(10,),
                dtype=float
            )
        })
        self.arm = BaseDorisArm()
        self.rgb_camera = BaseDorisCamera()
        self.depth_camera = BaseDorisCamera()
        self.wrist_rgb_camera = BaseDorisCamera()
        self.gripper = BaseDorisGripper()
        self.mobile_base = BaseDorisMobileBase()
        self.neck = BaseDorisNeck()
        self.joystick = BaseJoystick()
        self.rate = rospy.Rate(50.0)
        self.last_action = np.zeros(shape=self.action_space.shape)
        self.arm_pose = np.array([0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    def get_image_primary(self)->np.ndarray:
        return self.rgb_camera.get_camera_img()

    def get_image_wrist(self)->np.ndarray:
        return self.wrist_rgb_camera.get_camera_img()
    
    def get_depth_primary(self)->np.ndarray:
        return self.depth_camera.get_camera_img()
    
    def get_proprio(self)->np.ndarray:
        return np.concatenate([
            self.arm.get_arm_joints(),
            np.array([self.gripper.get_gripper_opening()]),
            self.mobile_base.get_mobile_base_vel(),
            self.neck.get_angles()
        ])
    
    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.get_image_primary()
    
    def get_obs(self)->dict[str, np.ndarray]:
        return {
            "pixels": {
                "top": self.get_image_primary(),
                "wrist": self.get_image_wrist(),
            },
            #"depth_primary": self.get_depth_primary(),
            "agent_pos": self.get_proprio()
        }
    
    def step(self, action: np.ndarray) -> Tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        #action[0] = -1.0*action[0]
        action[6] = 0.0
        action[7] = 0.0
        self.arm.set_arm_joints(action[:5])
        self.gripper.set_gripper_opening(action[5])
        #self.mobile_base.set_mobile_base_pose(action[7:10])
        if abs(action[6]) < 0.5:
            action[6] = 0.0
        if abs(action[7]) < 0.5:
            action[7] = 0.0
        self.mobile_base.set_mobile_base_vel(action[6:8])
        self.neck.set_angles(action[8:10])
        self.rate.sleep()
        #self.mobile_base.set_mobile_base_vel(np.array([0.0, 0.0]))
        return self.get_obs(), 0.0, False, False, {}
    
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[dict[str, np.ndarray] | dict]:
        self.arm_pose = np.array([0.3, 0.0, 0.3, 0.0, 0.0, 0.0])
        self.arm.set_arm_joints(self.arm.compute_ik(self.arm_pose))
        self.gripper.set_gripper_opening(0.1)
        self.neck.set_angles(np.array([0.0, 0.0]))
        self.mobile_base.set_mobile_base_vel(np.array([0.0, 0.0]))
        rospy.Rate(0.5).sleep()
        return self.get_obs(), {}
    
    def teleop(self):
        self.joystick.init()
        self.controller_state = 'base_neck'
        def act():
            action = self.last_action.copy()
            if self.joystick.get_axis_value(axis=2) == 1.0:
                self.controller_state = "base_neck"
            if self.joystick.get_axis_value(axis=5) == 1.0:
                self.controller_state = "arm_gripper"
            if self.controller_state == 'arm_gripper':
                new_arm_pose = self.arm_pose.copy()
                new_arm_pose[0] += 0.01 * self.joystick.get_axis_value(axis=1)
                new_arm_pose[1] += 0.01 * self.joystick.get_axis_value(axis=0)
                new_arm_pose[2] += 0.01 * self.joystick.get_axis_value(axis=4)
                new_arm_pose[3] += 0.01 * self.joystick.get_axis_value(axis=3)
                new_arm_pose[4] += 0.01 * self.joystick.get_axis_value(axis=7)
                new_arm_pose[5] = math.atan2(new_arm_pose[1], new_arm_pose[0])
                new_arm_pose = np.clip(new_arm_pose, np.array([-0.8, -0.8, -0.8, -2*np.pi, -2*np.pi, -2*np.pi]), np.array([0.8, 0.8, 0.8, 2*np.pi, 2*np.pi, 2*np.pi]))
                new_joints = self.arm.compute_ik(self.arm_pose)
                action[:5] = new_joints
                self.arm_pose = new_arm_pose
                if self.joystick.get_button_value(button=0):
                    action[5] += 0.01
                if self.joystick.get_button_value(button=1):
                    action[5] -= 0.01
            if self.controller_state == 'base_neck':
                angular = self.joystick.get_axis_value(axis=0)
                linear = self.joystick.get_axis_value(axis=1)
                if abs(linear) > abs(angular) + 0.05:
                    action[6] = linear
                    action[7] = 0.0
                elif abs(angular) > abs(linear) + 0.05:
                    action[6] = 0.0
                    action[7] = angular
                else:
                    action[6] = 0.0
                    action[7] = 0.0
                action[8] += 0.05 * self.joystick.get_axis_value(axis=3)
                action[9] += 0.05 * self.joystick.get_axis_value(axis=4)
            else:
                action[6] = 0.0
                action[7] = 0.0
            action = np.clip(action, self.action_min, self.action_max)
            if np.allclose(self.last_action, action) and action[6] == 0.0 and action[7] == 0.0:
                self.last_action = action
                return action, False
            else:
                self.last_action = action
                return action, True
        TeleopAgent = namedtuple('TeleopAgent', ['act'])
        agent = TeleopAgent(act=lambda: act())
        return agent
