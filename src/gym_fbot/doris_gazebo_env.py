from gym_fbot.doris_env import DorisEnv
from gym_fbot.arm.moveit_arm import MoveItArm
from gym_fbot.camera.rgb_camera import RGBCamera
from gym_fbot.camera.depth_camera import DepthCamera
from gym_fbot.gripper.moveit_gripper import MoveItGripper
from gym_fbot.mobile_base.navstack_mobile_base import NavStackMobileBase
from gym_fbot.neck.sim_neck import SimNeck
from gym_fbot.joystick.ros_joystick import ROSJoystick
import rospy
import curses
import h5py
import os
import numpy as np
from threading import Thread, Event
import cv2

class DorisGazeboEnv(DorisEnv):
    def __init__(self, **kwargs) -> None:
        try:
            rospy.init_node('doris_gazebo_env', anonymous=True)
        except rospy.ROSException:
            pass
        super(DorisGazeboEnv, self).__init__()
        self.arm = MoveItArm()
        self.rgb_camera = RGBCamera(topic="/camera/camera/color/image_raw")
        #self.wrist_rgb_camera = RGBCamera(topic="/doris_arm/camera/color/image_raw")
        #self.depth_camera = DepthCamera()
        self.gripper = MoveItGripper()
        #self.mobile_base = NavStackMobileBase()
        #self.neck = SimNeck()
        self.joystick = ROSJoystick()

dataset_path = '/home/cris/catkin_ws/src/fbot_gym/data/aloha_doris_gazebo_organize_shelf_raw'

def save_to_dataset(data: list, episode_id: int):
    path = f'{dataset_path}/episode_{episode_id}.hdf5'
    actions = np.array([d['action'] for d in data])
    qpos = np.array([d['obs']['agent_pos'] for d in data])
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    image_primary = [cv2.imencode('.jpg', d['obs']['pixels']['top'], encode_params)[1] for d in data]
    max_len_primary = max(len(img) for img in image_primary)
    #image_wrist = [cv2.imencode('.jpg', d['obs']['pixels']['wrist'], encode_params)[1] for d in data]
    #max_len_wrist = max(len(img) for img in image_wrist)
    with h5py.File(path, 'w') as data:
        data.create_dataset('/action', shape=actions.shape, dtype=actions.dtype, data=actions)
        data.create_dataset('/observations/qpos', shape=qpos.shape, dtype=qpos.dtype, data=qpos)
        #TODO: get joint velocities from the robot
        data.create_dataset('/observations/qvel', shape=qpos.shape, dtype=qpos.dtype, data=np.zeros_like(qpos))
        data.create_dataset('/observations/images/top', shape=(len(image_primary), max_len_primary), dtype=np.uint8)
        #data.create_dataset('/observations/images/wrist', shape=(len(image_wrist), max_len_wrist), dtype=np.uint8)
        for i, img in enumerate(image_primary):
            data['/observations/images/top'][i,:len(img)] = img
        #for i, img in enumerate(image_wrist):
        #    data['/observations/images/wrist'][i,:len(img)] = img
key = ''

def main(stdscr):
    global key
    rospy.init_node('doris_gazebo_env', anonymous=True)
    os.makedirs(dataset_path, exist_ok=True)
    env = DorisGazeboEnv()
    teleop_agent = env.teleop()
    state = 'IDLE'
    demo_id = len(os.listdir(dataset_path))
    buffer = []
    obs, _ = env.reset()
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
    while not rospy.is_shutdown():
        if key.lower() == 'q':
            raise KeyboardInterrupt()
        stdscr.clear()
        stdscr.addstr(0, 0, f'demo_id: {demo_id}, state: {state}')
        stdscr.refresh()
        action, changed = teleop_agent.act()
        if state == 'IDLE':
            if key.lower() == 'r':
                state = 'RECORD'
        if state == 'RECORD':
            if changed == True:
                buffer.append({
                    'obs': obs,
                    'action': action,
                })
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
        if changed == True:
            obs, _, _, _, _ = env.step(action)
        cv2.imshow('Top Camera', cv2.cvtColor(obs['pixels']['top'], cv2.COLOR_RGB2BGR))
        #cv2.imshow('Wrist Camera', cv2.cvtColor(obs['pixels']['wrist'], cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

if __name__ == '__main__':
    curses.wrapper(main)
