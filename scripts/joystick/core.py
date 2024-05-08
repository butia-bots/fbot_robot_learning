class BaseJoystick:
    def get_axis_value(self, axis: int)->float:
        raise NotImplementedError()
    
    def get_button_value(self, button: int)->bool:
        raise NotImplementedError()