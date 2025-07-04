from pyfame.timing.timing_curves import timing_linear
from pyfame.mesh.get_mesh_landmarks import FACE_OVAL_PATH
from pyfame.util.util_constants import OCCLUSION_FILL_BLACK
from pyfame.manipulation.occlusion.layer_occlusion_path import LayerOcclusionPath
from typing import Callable

def get_layer_occlusion_path(fill_method:int|str = OCCLUSION_FILL_BLACK, onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, 
                 roi:list[list[tuple]] = [FACE_OVAL_PATH], fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerOcclusionPath:
    
    return LayerOcclusionPath(fill_method, onset_t, offset_t, timing_func, roi, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)