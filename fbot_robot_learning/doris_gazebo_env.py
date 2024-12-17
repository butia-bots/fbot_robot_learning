#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from fbot_robot_learning.doris_env import DorisEnv
from fbot_robot_learning.arm.moveit_arm import MoveItArm
from fbot_robot_learning.camera.rgb_camera import RGBCamera
from fbot_robot_learning.gripper.moveit_gripper import MoveItGripper
from fbot_robot_learning.joystick.ros_joystick import ROSJoystick
import curses
import h5py
import os
import numpy as np
from threading import Thread, Event
import cv2

class DorisGazeboEnv(DorisEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.arm = MoveItArm()
        self.rgb_camera = RGBCamera(topic="/camera/camera/color/image_raw")
        self.gripper = MoveItGripper()
        self.joystick = ROSJoystick()

dataset_path = '/home/cris/catkin_ws/src/fbot_gym/data/aloha_doris_gazebo_organize_shelf_raw'

def save_to_dataset(data: list, episode_id: int):
    path = f'{dataset_path}/episode_{episode_id}.hdf5'
    actions = np.array([d['action'] for d in data])
    qpos = np.array([d['obs']['agent_pos'] for d in data])
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    image_primary = [cv2.imencode('.jpg', d['obs']['pixels']['top'], encode_params)[1] for d in data]
    max_len_primary = max(len(img) for img in image_primary)
    with h5py.File(path, 'w') as data_file:
        data_file.create_dataset('/action', shape=actions.shape, dtype=actions.dtype, data=actions)
        data_file.create_dataset('/observations/qpos', shape=qpos.shape, dtype=qpos.dtype, data=qpos)
        data_file.create_dataset('/observations/qvel', shape=qpos.shape, dtype=qpos.dtype, data=np.zeros_like(qpos))
        data_file.create_dataset('/observations/images/top', shape=(len(image_primary), max_len_primary), dtype=np.uint8)
        for i, img in enumerate(image_primary):
            data_file['/observations/images/top'][i, :len(img)] = img

class TeleopAgent:
    def __init__(self, env):
        self.env = env

    def act(self):
        # Add logic to define agent actions here.
        pass

def inner_main(stdscr):
    # Initialize ROS2
    rclpy.init()
    os.makedirs(dataset_path, exist_ok=True)
    env = DorisGazeboEnv()
    teleop_agent = TeleopAgent(env)
    state = 'IDLE'
    demo_id = len(os.listdir(dataset_path))
    buffer = []
    obs, _ = env.reset()  # Ensure your environment has a reset() method
    key_pressed = Event()

    def detect_key_press():
        global key
        while True:
            key_pressed.clear()
            key = stdscr.getkey()
            key_pressed.set()

    thread = Thread(target=detect_key_press)
    thread.start()

    action = None
    while rclpy.ok():
        if key.lower() == 'q':
            raise KeyboardInterrupt()
        stdscr.clear()
        stdscr.addstr(0, 0, f'demo_id: {demo_id}, state: {state}')
        stdscr.refresh()

        # Teleoperation action step
        action, changed = teleop_agent.act()

        if state == 'IDLE' and key.lower() == 'r':
            state = 'RECORD'
        
        if state == 'RECORD':
            if changed:
                buffer.append({'obs': obs, 'action': action})
            if key.lower() == 's':
                save_to_dataset(data=buffer, episode_id=demo_id)
                buffer.clear()
                demo_id += 1
                obs, _ = env.reset()
                state = 'IDLE'
            if key.lower() == 'd':
                buffer.clear()
                obs, _ = env.reset()
                state = 'IDLE'

        if changed:
            obs, _, _, _, _ = env.step(action)  # Ensure your environment has a step() method

        cv2.imshow('Top Camera', cv2.cvtColor(obs['pixels']['top'], cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    rclpy.shutdown()

def main(args=None):
    curses.wrapper(inner_main)
