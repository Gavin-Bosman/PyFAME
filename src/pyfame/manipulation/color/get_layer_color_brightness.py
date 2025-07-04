from pyfame.timing.timing_curves import timing_linear
from pyfame.mesh.get_mesh_landmarks import FACE_OVAL_PATH
from pyfame.manipulation.color.layer_color_brightness import LayerColorBrightness
from typing import Callable

def get_layer_color_brightness(onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, roi:list[list[tuple]] = [FACE_OVAL_PATH], 
                 fade_duration:int = 500, magnitude:float = 20.0, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerColorBrightness:
    
    return LayerColorBrightness(onset_t, offset_t, timing_func, roi, fade_duration, magnitude, min_tracking_confidence, min_detection_confidence, **kwargs)