from pyfame.util.util_constants import *
from pyfame.mesh import get_mask_from_path
from pyfame.layer import Layer
from pyfame.timing.timing_curves import timing_linear
from pyfame.mesh.get_mesh_landmarks import FACE_OVAL_PATH
from pyfame.util.util_exceptions import *
from pyfame.util.util_checks import *
import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerColor(Layer):

    def __init__(self, onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, 
                 roi:list[list[tuple]] = [FACE_OVAL_PATH], fade_duration:int = 500, focus_color:str|int = "red", 
                 magnitude:float = 10.0, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        super().__init__(onset_t, offset_t, timing_func, roi, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        check_type(focus_color, [str,int])
        check_value(focus_color, ["red", "green", "blue", "yellow", 4, 5, 6, 7])
        check_type(magnitude, [float])
        check_value(magnitude, min=0)

        self.roi = roi
        self.magnitude = magnitude
        self.color = focus_color
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.static_image_mode = False
    
    def supports_weight(self) -> bool:
        return True
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):
        weight = None

        if self.static_image_mode != static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        if self.static_image_mode:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())
        
        if weight == 0.0:
            return frame
        else:
            face_mesh = super().get_face_mesh()
            mask = get_mask_from_path(frame, self.roi, face_mesh)

            # Convert input image to CIE La*b* color space (perceptually uniform space)
            img_LAB = cv.cvtColor(frame, cv.COLOR_BGR2LAB).astype(np.float32)
            l,a,b = cv.split(img_LAB)

            match self.color:
                case "red" | 4:
                    a = np.where(mask==255, a + (weight * self.magnitude), a)
                    np.clip(a, -128, 127)
                case "blue" | 5:
                    b = np.where(mask==255, b - (weight * self.magnitude), b)
                    np.clip(b, -128, 127)
                case "green" | 6:
                    a = np.where(mask==255, a - (weight * self.magnitude), a)
                    np.clip(a, -128, 127)
                case "yellow" | 7:
                    b = np.where(mask==255, b + (weight * self.magnitude), b)
                    np.clip(b, -128, 127)
            
            img_LAB = cv.merge([l,a,b])

            # Convert CIE La*b* back to BGR
            result = cv.cvtColor(img_LAB.astype(np.uint8), cv.COLOR_LAB2BGR)
            return result