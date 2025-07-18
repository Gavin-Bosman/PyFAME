from pyfame.utilities.util_constants import *
from pyfame.mesh import *
from pyfame.utilities.util_checks import *
from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerOcclusionBlur(Layer):
    def __init__(self, method:str|int = "gaussian", kernel_size:int = 15, onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, 
                 roi:list[list[tuple]] = [FACE_OVAL_PATH], fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        super().__init__(onset_t, offset_t, timing_func, roi, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        check_type(method, [str, int])
        check_value(method, [11,12,13,"average","gaussian","median"])
        check_type(kernel_size, [int])
        check_value(kernel_size, min=5)

        
        self.method = method
        self.k_size = kernel_size
        self.roi = roi
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.static_image_mode = False
    
    def supports_weight(self):
        return False
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):

        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            face_mesh = super().get_face_mesh()
            mask = get_mask_from_path(frame, self.roi, face_mesh)
            output_frame = np.zeros_like(frame, dtype=np.uint8)

            match self.method:
                case "average" | 11:
                    frame_blurred = cv.blur(frame, (self.k_size, self.k_size))
                    output_frame = np.where(mask == 255, frame_blurred, frame)
                
                case "gaussian" | 12:
                    frame_blurred = cv.GaussianBlur(frame, (self.k_size, self.k_size), 0)
                    output_frame = np.where(mask == 255, frame_blurred, frame)
                
                case "median" | 13:
                    frame_blurred = cv.medianBlur(frame, self.k_size)
                    output_frame = np.where(mask == 255, frame_blurred, frame)
            
            return output_frame

def layer_occlusion_blur(time_onset:float=None, time_offset:float=None, timing_function:Callable[...,float]=timing_linear, method:str|int = "gaussian", kernel_size:int = 15,
                 region_of_interest:list[list[tuple]] = [FACE_OVAL_PATH], fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerOcclusionBlur:
    
    return LayerOcclusionBlur(method, kernel_size, time_onset, time_offset, timing_function, region_of_interest, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)