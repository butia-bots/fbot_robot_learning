from gripper.core import BaseDorisGripper
from moveit_commander.move_group import MoveGroupCommander

class MoveItGripper(BaseDorisGripper):
    def __init__(self) -> None:
        super(MoveItGripper, self).__init__()
        self.move_group = MoveGroupCommander('gripper', robot_description='doris_arm/robot_description', ns='/doris_arm')

    def set_gripper_opening(self, opening: float):
        self.move_group.stop()
        self.move_group.set_joint_value_target([opening/2.0,-opening/2.0])
        self.move_group.go(wait=False)
    
    def get_gripper_opening(self) -> float:
        return sum([abs(j) for j in self.move_group.get_current_joint_values()])