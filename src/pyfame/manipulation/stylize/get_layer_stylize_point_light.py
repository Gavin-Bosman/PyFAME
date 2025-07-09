from pyfame.manipulation.stylize.layer_stylize_point_light import LayerStylizePointLight
from pyfame.timing.timing_curves import timing_linear
from pyfame.mesh.get_mesh_landmarks import FACE_OVAL_PATH
from pyfame.util.util_constants import SHOW_HISTORY_ORIGIN
from typing import Callable

def get_layer_stylize_point_light(point_density:float = 1.0, point_color:tuple[int] = (255,255,255), maintain_background:bool = True, display_history_vectors:bool = False, 
                 history_method:int|str = SHOW_HISTORY_ORIGIN, history_window_msec:int = 500, history_vec_color:tuple[int] = (0,0,255), onset_t:float=None, 
                 offset_t:float=None, timing_func:Callable[...,float]=timing_linear, roi:list[list[tuple]] = [FACE_OVAL_PATH], fade_duration:int = 500, 
                 min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
    
    return LayerStylizePointLight(point_density, point_color, maintain_background, display_history_vectors, history_method, history_window_msec, 
                                  history_vec_color, onset_t, offset_t, timing_func, roi, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)