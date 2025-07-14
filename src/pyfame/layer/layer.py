from typing import Callable
from abc import ABC, abstractmethod
from cv2.typing import MatLike
from pyfame.layer.timing_curves import timing_linear
from pyfame.utilities.util_checks import *
from pyfame.mesh.get_mesh_landmarks import FACE_OVAL_PATH
from pyfame.mesh.get_mesh_coordinates import get_mesh

class Layer(ABC): 
    """ An abstract base class to be extended by pyfame's manipulation layer classes. """

    def __init__(self, onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, roi:list[list[tuple]] = [FACE_OVAL_PATH], 
                 fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        
        check_type(onset_t, [float, type(None)])
        check_type(offset_t, [float, type(None)])
        check_type(fade_duration, [int])
        check_value(fade_duration, min=0)

        self.face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, False)
        self.onset_t = onset_t
        self.offset_t = offset_t
        self.timing_func = timing_func
        self.roi = roi
        self.fade_duration = fade_duration/1000
        self.timing_kwargs = kwargs
    
    def compute_weight(self, dt:float, supports_weight:bool) -> float:
        if self.onset_t and self.offset_t:
            if dt < self.onset_t:
                return 0.0
            elif self.onset_t <= dt <= self.offset_t:
                if supports_weight:
                    return self.timing_func(dt, self.onset_t, self.offset_t, **self.timing_kwargs)
                else:
                    return 1.0
            elif self.offset_t <= dt <= self.offset_t + self.fade_duration:
                if supports_weight:
                    fade_t = (dt - self.offset_t)/self.fade_duration
                    return self.timing_func(dt, self.onset_t, self.offset_t, **self.timing_kwargs) * (1-fade_t)
                else:
                    return 0.0
            else:
                return 0.0
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
    def apply_layer(self, frame:MatLike, dt:float, static_image_mode:bool = False) -> MatLike:
        pass
