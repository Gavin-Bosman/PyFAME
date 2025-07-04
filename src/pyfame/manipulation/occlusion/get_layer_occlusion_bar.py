from pyfame.timing.timing_curves import timing_linear
from pyfame.mesh.get_mesh_landmarks import FACE_OVAL_PATH
from pyfame.manipulation.occlusion.layer_occlusion_bar import LayerOcclusionBar
from typing import Callable

def get_layer_occlusion_bar(bar_color:tuple[int,int,int] = (0,0,0), onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, 
                 roi:list[list[tuple]] = [FACE_OVAL_PATH], fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerOcclusionBar:
    
    return LayerOcclusionBar(bar_color, onset_t, offset_t, timing_func, roi, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)