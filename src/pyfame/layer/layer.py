from typing import Callable
from abc import ABC, abstractmethod
from cv2.typing import MatLike
from pyfame.layer.timing_curves import timing_linear
from pyfame.utilities.checks import *
from pyfame.mesh.get_mesh_coordinates import get_mesh


class Layer(ABC): 
    """ An abstract base class to be extended by pyfame's manipulation layer classes. """

    def __init__(self, onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, rise_duration:int = 500, 
                 fall_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        
        check_type(onset_t, [float, type(None)])
        check_type(offset_t, [float, type(None)])
        check_type(fall_duration, [int])
        check_value(fall_duration, min=0)
        check_type(rise_duration, [int])
        check_value(rise_duration, min=0)
        check_type(min_detection_confidence, [float])
        check_value(min_detection_confidence, min=0.0, max=1.0)
        check_type(min_tracking_confidence, [float])
        check_value(min_tracking_confidence, min=0.0, max=1.0)

        self.face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, False)
        self.onset_t = onset_t
        self.offset_t = offset_t
        self.timing = timing_func
        self.rise = rise_duration/1000
        self.fall = fall_duration/1000
        self.time_kwargs = kwargs
        
    def compute_weight(self, dt:float, supports_weight:bool) -> float:
        if supports_weight:
            return self.timing(dt, self.onset_t, self.offset_t, self.rise, self.fall, **self.time_kwargs)
        else:
            return 1.0
    
    def get_face_mesh(self):
        return self.face_mesh
    
    def set_face_mesh(self, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, static_image_mode:bool = False):
        self.face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, static_image_mode)

    @abstractmethod
    def supports_weight(self) -> bool:
        pass

    @abstractmethod
    def get_layer_parameters(self) -> dict:
        pass

    @abstractmethod
    def apply_layer(self, frame:MatLike, dt:float, static_image_mode:bool = False) -> MatLike:
        pass
