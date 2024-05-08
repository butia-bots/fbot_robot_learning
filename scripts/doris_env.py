from typing import List, Tuple
from gym import Env
from gym.spaces import Box, Dict
import numpy as np
from arm.core import BaseDorisArm
from camera.core import BaseDorisCamera
from gripper.core import BaseDorisGripper
from mobile_base.core import BaseDorisMobileBase
from neck.core import BaseDorisNeck
from joystick.core import BaseJoystick
from collections import namedtuple
import pygame
import math
import rospy


class DorisEnv(Env):
    def __init__(self) -> None:
        #(ee_x, ee_y, ee_z, ee_roll, ee_pitch, ee_yaw, gripper_opening, base_x, base_y, base_yaw, neck_horizontal, neck_vertical)
        self.action_min = np.array([-1.0,-1.0,-1.0,-2*np.pi,-2*np.pi,-2*np.pi,0.0,-100.0,-100.0,-2*np.pi,-2*np.pi,-2*np.pi], dtype=float)
        self.action_max = np.array([1.0,1.0,1.0,2*np.pi,2*np.pi,2*np.pi,0.0,100.0,100.0,2*np.pi,2*np.pi,2*np.pi], dtype=float)
        self.action_space = Box(
            low=self.action_min,
            high=self.action_max,
            shape=(12,),
            dtype=float
        )
        self.observation_space = Dict({
            "image_primary": Box(
                low=0,
                high=255,
                shape=(480, 640, 3),
                dtype=np.uint8
            ),
            "depth_primary": Box(
                low=0,
                high=255,
                shape=(480, 640, 1),
                dtype=np.uint8
            ),
            #(ee_x, ee_y, ee_z, ee_roll, ee_pitch, ee_yaw, gripper_opening, base_x, base_y, base_yaw, neck_horizontal, neck_vertical)
            "proprio": Box(
                low=np.array([-1.0,-1.0,-1.0,-2*np.pi,-2*np.pi,-2*np.pi,0.0,-100.0,-100.0,-2*np.pi,-2*np.pi,-2*np.pi], dtype=float),
                high=np.array([1.0,1.0,1.0,2*np.pi,2*np.pi,2*np.pi,0.0,100.0,100.0,2*np.pi,2*np.pi,2*np.pi], dtype=float),
                shape=(12,),
                dtype=float
            )
        })
        self.arm = BaseDorisArm()
        self.rgb_camera = BaseDorisCamera()
        self.depth_camera = BaseDorisCamera()
        self.gripper = BaseDorisGripper()
        self.mobile_base = BaseDorisMobileBase()
        self.neck = BaseDorisNeck()
        self.joystick = BaseJoystick()
        self.rate = rospy.Rate(10.0)
    
    def get_image_primary(self)->np.ndarray:
        return self.rgb_camera.get_camera_img()
    
    def get_depth_primary(self)->np.ndarray:
        return self.depth_camera.get_camera_img()
    
    def get_proprio(self)->np.ndarray:
        return np.concatenate([
            self.arm.get_arm_pose(),
            np.array(self.gripper.get_gripper_opening()),
            self.mobile_base.get_mobile_base_pose(),
            self.neck.get_angles()
        ])
    
    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.get_image_primary()
    
    def get_obs(self)->dict[str, np.ndarray]:
        return {
            "image_primary": self.get_image_primary(),
            "depth_primary": self.get_depth_primary(),
            "proprio": self.get_proprio()
        }
    
    def step(self, action: np.ndarray) -> Tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        self.arm.set_arm_pose(action[:6])
        self.gripper.set_gripper_opening(action[6])
        self.mobile_base.set_mobile_base_pose(action[7:10])
        self.neck.set_angles(action[10:12])
        self.rate.sleep()
        return self.get_obs(), 0.0, False, False, {}
    
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[dict[str, np.ndarray] | dict]:
        return self.get_obs(), {}
    
    def teleop(self):
        self.controller_state = 'base_neck'
        def act():
            action = self.get_proprio()
            if self.joystick.get_button_value(button=pygame.CONTROLLER_BUTTON_LEFTSHOULDER):
                self.controller_state = "base_neck"
            if self.joystick.get_button_value(button=pygame.CONTROLLER_BUTTON_RIGHTSHOULDER):
                self.controller_state = "arm_gripper"
            if self.controller_state == 'arm_gripper':
                action[0] += 0.05 * self.joystick.get_axis_value(axis=pygame.CONTROLLER_AXIS_LEFTX)
                action[1] += 0.05 * self.joystick.get_axis_value(axis=pygame.CONTROLLER_AXIS_LEFTY)
                action[2] += 0.05 * self.joystick.get_axis_value(axis=pygame.CONTROLLER_AXIS_RIGHTY)
                action[3] += 0.05 * self.joystick.get_axis_value(axis=pygame.CONTROLLER_AXIS_RIGHTX)
                if self.joystick.get_button_value(button=pygame.CONTROLLER_BUTTON_DPAD_UP):
                    action[4] += 0.05
                if self.joystick.get_button_value(button=pygame.CONTROLLER_BUTTON_DPAD_DOWN):
                    action[4] -= 0.05
                action[5] = math.atan2(action[1], action[0])
                if self.joystick.get_button_value(button=pygame.CONTROLLER_BUTTON_A):
                    action[6] += 0.05
                if self.joystick.get_button_value(button=pygame.CONTROLLER_BUTTON_B):
                    action[6] -= 0.05
            if self.controller_state == 'base_neck':
                action[9] += 0.05 * self.joystick.get_axis_value(axis=pygame.CONTROLLER_AXIS_LEFTY)
                action[9] = action[9] % 2*np.pi
                linear = self.joystick.get_axis_value(axis=pygame.CONTROLLER_AXIS_LEFTX)
                action[7] += 0.05 * linear * math.cos(action[9])
                action[8] += 0.05 * linear * math.sin(action[9])
                action[10] += 0.05 * self.joystick.get_axis_value(axis=pygame.CONTROLLER_AXIS_RIGHTX)
                action[11] += 0.05 * self.joystick.get_axis_value(axis=pygame.CONTROLLER_AXIS_RIGHTY)
            action = np.clip(action, self.action_min, self.action_max)
            return action
        TeleopAgent = namedtuple('TeleopAgent', ['act'])
        agent = TeleopAgent(act=lambda: act())
        return agent