from gym_fbot.joystick.core import BaseJoystick
from sensor_msgs.msg import Joy
import rospy

class ROSJoystick(BaseJoystick):
    def __init__(self) -> None:
        super().__init__()
        self.joy = rospy.Subscriber('/joy', Joy, self._update_joy_msg)

    def _update_joy_msg(self, msg: Joy):
        self.joy_msg = msg

    def get_axis_value(self, axis: int) -> float:
        #rospy.wait_for_message('/joy', Joy)
        return self.joy_msg.axes[axis]
    
    def get_button_value(self, button: int) -> bool:
        return self.joy_msg.buttons[button]

    def init(self):
        rospy.wait_for_message('/joy', Joy)
