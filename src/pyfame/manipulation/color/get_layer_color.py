from pyfame.timing.timing_curves import timing_linear
from pyfame.mesh.get_mesh_landmarks import FACE_OVAL_PATH
from pyfame.manipulation.color.layer_color import LayerColor
from typing import Callable

def get_layer_color(onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, 
                 roi:list[list[tuple]] = [FACE_OVAL_PATH], fade_duration:int = 500, focus_color:str|int = "red", 
                 magnitude:float = 10.0, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerColor:
    
    return LayerColor(onset_t, offset_t, timing_func, roi, fade_duration, focus_color, magnitude, min_tracking_confidence, min_detection_confidence, **kwargs)