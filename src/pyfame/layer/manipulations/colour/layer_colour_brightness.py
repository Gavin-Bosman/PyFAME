from pyfame.utilities.constants import *
from pyfame.layer.manipulations.mask import mask_from_path
from pyfame.mesh.mesh_landmarks import FACE_OVAL_PATH
from pyfame.layer.layer import Layer
from pyfame.utilities.checks import *
from pyfame.layer.timing_curves import timing_linear
from pyfame.utilities.general_utilities import sanitize_json_value, get_roi_name
import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerColourBrightness(Layer):
    def __init__(self, onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, roi:list[list[tuple]] | list[tuple] = FACE_OVAL_PATH, 
                 rise_duration:int=500, fall_duration:int = 500, magnitude:float = 20.0, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        # Initialise the superclass
        super().__init__(onset_t, offset_t, timing_func, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        self.static_image_mode = False
        self._pre_attrs = []
        self._pre_attrs = set(self.__dict__) # snapshot of just the superclass parameters
        
        # Perform parameter checks
        check_type(magnitude, [float])

        # Define class parameters
        self.time_onset = onset_t
        self.time_offset = offset_t
        self.timing_function = timing_func
        self.region_of_interest = roi
        self.rise_duration_msec = rise_duration
        self.fall_duration_msec = fall_duration
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
    
    def supports_weight(self):
        return True
    
    def get_layer_parameters(self) -> dict:
        return dict(self._layer_parameters)

    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):
        weight = None

        # Update the faceMesh when switching between image and video processing
        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        if self.static_image_mode:
            weight = 1.0    
        else:
            weight = super().compute_weight(dt, self.supports_weight())
        
        # Occurs when the dt < onset_time, or > offset_time
        if weight == 0.0:
            return frame
        else:
            # Mask out the region of interest
            face_mesh = super().get_face_mesh()
            mask = mask_from_path(frame, self.region_of_interest, face_mesh)
            # Reshape the mask for compatibility with cv2.convertScaleAbs()
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

            # Otsu thresholding to seperate foreground and background
            grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
            thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

            # Adding a temporary image border to allow for correct floodfill behaviour
            bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
            floodfilled = bordered_thresholded.copy()
            cv.floodFill(floodfilled, None, (0,0), 255)

            # Removing temporary border and creating foreground mask
            floodfilled = floodfilled[10:-10, 10:-10]
            floodfilled = cv.bitwise_not(floodfilled)
            foreground = cv.bitwise_or(thresholded, floodfilled)

            # Within the masked region, upscale the brightness according to the current weight
            img_brightened = np.where(mask == 255, cv.convertScaleAbs(src=frame, alpha=1, beta=(weight * self.magnitude)), frame)
            img_brightened[foreground == 0] = frame[foreground == 0]
            return img_brightened

def layer_colour_brightness(time_onset:float=None, time_offset:float=None, timing_function:Callable[...,float]=timing_linear, region_of_interest:list[list[tuple]] | list[tuple] = FACE_OVAL_PATH, 
                rise_duration:int=500,fall_duration:int=500, magnitude:float = 20.0, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerColourBrightness:
    
    return LayerColourBrightness(time_onset, time_offset, timing_function, region_of_interest, rise_duration, fall_duration, magnitude, min_tracking_confidence, min_detection_confidence, **kwargs)