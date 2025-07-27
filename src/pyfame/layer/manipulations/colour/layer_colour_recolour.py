from pyfame.utilities.constants import *
from pyfame.layer.manipulations.mask import mask_from_path
from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
from pyfame.mesh.mesh_landmarks import FACE_OVAL_PATH
from pyfame.utilities.exceptions import *
from pyfame.utilities.checks import *
from pyfame.utilities.general_utilities import sanitize_json_value, get_roi_name
import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerColourRecolour(Layer):

    def __init__(self, onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, roi:list[list[tuple]]|list[tuple] = FACE_OVAL_PATH, 
                 rise_duration:int = 500, fall_duration:int = 500, focus_color:str|int = "red", magnitude:float = 10.0, min_tracking_confidence:float = 0.5, 
                 min_detection_confidence:float = 0.5, **kwargs):
        # Initialise the superclass
        super().__init__(onset_t, offset_t, timing_func, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        self.static_image_mode = False
        self._pre_attrs = []
        self._pre_attrs = set(self.__dict__) # snapshot of just the superclass parameters
        
        # Perform parameter checks
        check_type(focus_color, [str,int])
        check_value(focus_color, ["red", "green", "blue", "yellow", 4, 5, 6, 7])
        check_type(magnitude, [float])
        check_value(magnitude, min=0)
        
        # Define class parameters
        self.time_onset = onset_t
        self.time_offset = offset_t
        self.timing_function = timing_func
        self.region_of_interest = roi
        self.rise_duration_msec = rise_duration
        self.fall_duration_msec = fall_duration
        self.focus_colour = focus_color
        self.magnitude = magnitude
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
    
    def supports_weight(self) -> bool:
        return True
    
    def get_layer_parameters(self) -> dict:
        return dict(self._layer_parameters)
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):
        weight = None

        # Update the faceMesh when switching between image and video processing
        if self.static_image_mode != static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        # Since a static image is essentially one frame, perform the manipulation at maximal weight
        if self.static_image_mode:
            weight = 1.0
        else:
            weight = super().compute_weight(dt, self.supports_weight())
        
        # Occurs when the current dt is less than the onset_time, or greater than the offset_time
        if weight == 0.0:
            return frame
        else:
            # Get a mask of our region of interest
            face_mesh = super().get_face_mesh()
            mask = mask_from_path(frame, self.region_of_interest, face_mesh)

            # Convert input image to CIE La*b* color space (perceptually uniform space)
            img_LAB = cv.cvtColor(frame, cv.COLOR_BGR2LAB).astype(np.float32)
            # Split the image into individual channels for precise colour manipulation
            l,a,b = cv.split(img_LAB)

            # Shift the various colour channels according to the user-specified focus_colour
            match self.focus_colour:
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
            
            # After shifting the colour channels, merge the individual channels back into one image
            img_LAB = cv.merge([l,a,b])

            # Convert CIE La*b* back to BGR
            result = cv.cvtColor(img_LAB.astype(np.uint8), cv.COLOR_LAB2BGR)
            return result

def layer_colour_recolour(time_onset:float=None, time_offset:float=None, timing_function:Callable[...,float]=timing_linear, 
                 region_of_interest:list[list[tuple]] | list[tuple] = FACE_OVAL_PATH, rise_duration:int = 500, fall_duration:int = 500, focus_colour:str|int = "red", 
                 magnitude:float = 10.0, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **timing_kwargs) -> LayerColourRecolour:
    
    return LayerColourRecolour(time_onset, time_offset, timing_function, region_of_interest, rise_duration, fall_duration, focus_colour, magnitude, min_tracking_confidence, min_detection_confidence, **timing_kwargs)