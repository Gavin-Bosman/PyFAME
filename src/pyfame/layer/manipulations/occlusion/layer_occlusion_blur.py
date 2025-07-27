from pyfame.utilities.constants import *
from pyfame.mesh import *
from pyfame.utilities.checks import *
from pyfame.utilities.general_utilities import sanitize_json_value, get_roi_name
from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
from pyfame.layer.manipulations.mask import mask_from_path
import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerOcclusionBlur(Layer):
    def __init__(self, method:str|int = "gaussian", kernel_size:int = 15, onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, 
                 roi:list[list[tuple]] | list[tuple] = FACE_OVAL_PATH, rise_duration:int=500, fall_duration:int=500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        # Initialise superclass
        super().__init__(onset_t, offset_t, timing_func, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        self.static_image_mode = False
        self._pre_attrs = []
        self._pre_attrs = set(self.__dict__) # snapshot of just the superclass parameters
        
        # Perform parameter checks
        check_type(method, [str, int])
        check_value(method, [11,12,13,"average","gaussian","median"])
        check_type(kernel_size, [int])
        check_value(kernel_size, min=5)

        # Define class parameters
        self.blur_method = method
        self.kernel_size = kernel_size
        self.time_onset = onset_t
        self.time_offset = offset_t
        self.timing_function = timing_func
        self.region_of_interest = roi
        self.rise_duration_msec = rise_duration
        self.fall_duration_msec = fall_duration
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.timing_kwargs = kwargs

        self._capture_init_params()
    
    def _capture_init_params(self):
        # Extracting total parameter list post init
        post_attrs = set(self.__dict__.keys())

        # Getting only the subclass parameters
        new_attrs = post_attrs - self._pre_attrs

        # Store only subclass level params; ignore self
        params = {attr: getattr(self, attr) for attr in new_attrs}

        # Handle non serializable types
        if "region_of_interest" in params:
            params["region_of_interest"] = get_roi_name(params["region_of_interest"])

        self._layer_parameters = {
            k: sanitize_json_value(v) for k, v in params.items()
        }
    
    def supports_weight(self):
        return False
    
    def get_layer_parameters(self):
        return dict(self._layer_parameters)
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):

        # Update faceMesh when switching between image and video processing
        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        # Blurring does not support weight, so weight will always be 0.0 or 1.0
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            # Mask out region of interest
            face_mesh = super().get_face_mesh()
            mask = mask_from_path(frame, self.region_of_interest, face_mesh)
            output_frame = np.zeros_like(frame, dtype=np.uint8)

            # Blur the input frame depending on user-specified blur method
            match self.blur_method:
                case "average" | 11:
                    frame_blurred = cv.blur(frame, (self.kernel_size, self.kernel_size))
                    output_frame = np.where(mask == 255, frame_blurred, frame)
                
                case "gaussian" | 12:
                    frame_blurred = cv.GaussianBlur(frame, (self.kernel_size, self.kernel_size), 0)
                    output_frame = np.where(mask == 255, frame_blurred, frame)
                
                case "median" | 13:
                    frame_blurred = cv.medianBlur(frame, self.kernel_size)
                    output_frame = np.where(mask == 255, frame_blurred, frame)
            
            return output_frame

def layer_occlusion_blur(time_onset:float=None, time_offset:float=None, timing_function:Callable[...,float]=timing_linear, method:str|int = "gaussian", kernel_size:int = 15,
                 region_of_interest:list[list[tuple]] | list[tuple] = FACE_OVAL_PATH, rise_duration:int=500, fall_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerOcclusionBlur:
    
    return LayerOcclusionBlur(method, kernel_size, time_onset, time_offset, timing_function, region_of_interest, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)