# Install

Install python dependencies with following command

```sh
pip install -r requirements.txt
```

Install ROS joy package with following command

```sh
sudo apt install ros-noetic-joy
```

# Collect demonstrations dataset

Edit doris_gazebo_env.py so that the dataset_path variable points to the directory where you want to save your dataset (make sure the name of the directory starts with `aloha_`, ends with `_raw`, and do not have the words `sim` on it)

```python
# ...
dataset_path = '/home/cris/catkin_ws/src/fbot_gym/data/aloha_doris_gazebo_organize_shelf_raw'
# ...
```

Launch the robot's RosAria, MoveIt and Realsense launchfiles, and then run the joy_node to connect an Xbox 360 joystick to ROS

```sh
rosrun joy joy_node
```

Then run doris_gazebo_env.py in order to start collecting the dataset.

```sh
python src/doris_gazebo_env.py
```

Once doris_gazebo_env.py is running, wait a few seconds and press any button on the joystick in order to initialize the demo collection interface, built using curses. On the interface, press `R` on the keyboard to change state from `IDLE` to `RECORD` and begin recording actions and observations. Use the joystick to operate the robot to complete the task, and then press `S` on the keyboard to save the demonstration to the dataset. In case you commit some mistake when operating the robot (Eg.: knocking over some object) that could lead to sub-optimal behavior, press `D` to discard the demo. Make sure to randomize the poses of relevant objects between the demos.

# Controls

- Joystick left axis Y - Arm EE X axis
- Joystick left axis X - Arm EE Y axis
- Joystick right axis Y - Arm EE Z axis
- Joystick right axis X - Arm EE Roll
- Joystick D-PAD up and down - Arm EE Pitch
- Joystick `A` button - Open gripper
- Joystick `B` button - Close gripper