from pyfame.timing.timing_curves import timing_linear
from pyfame.mesh.get_mesh_landmarks import FACE_OVAL_PATH
from pyfame.manipulation.spatial.layer_spatial_grid_shuffle import LayerSpatialGridShuffle
from typing import Callable

def get_layer_spatial_grid_shuffle(rand_seed:int|None, method:int|str = "fully_random", shuffle_threshold:int = 2, grid_square_size:int = 40, 
                 onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, roi:list[list[tuple]] = [FACE_OVAL_PATH], 
                 fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerSpatialGridShuffle:
    
    return LayerSpatialGridShuffle(rand_seed, method, shuffle_threshold, grid_square_size, onset_t, offset_t, 
                                   timing_func, roi, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)