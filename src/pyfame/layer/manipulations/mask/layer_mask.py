from pyfame.utilities.constants import *
from pyfame.utilities.checks import *
from pyfame.utilities.general_utilities import get_roi_name, sanitize_json_value
from pyfame.layer.manipulations.mask import mask_from_path
from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
from pyfame.mesh.mesh_landmarks import FACE_OVAL_PATH
import numpy as np
import cv2 as cv
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerMask(Layer):
    def __init__(self, background_color:tuple[int,int,int] = (0,0,0), onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, 
                 roi:list[list[tuple]] | list[tuple] = FACE_OVAL_PATH, rise_duration:int=500, fall_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        # Initialise the superclass
        super().__init__(onset_t, offset_t, timing_func, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        self.static_image_mode = False
        self._pre_attrs = []
        self._pre_attrs = set(self.__dict__) # snapshot of just the superclass parameters
        
        # Perform parameter checks
        check_type(background_color, [tuple])
        check_type(background_color, [int], iterable=True)
        for i in background_color:
            check_value(i, min=0, max=255)

        # Define class parameters
        self.time_onset = onset_t
        self.time_offset = offset_t
        self.timing_function = timing_func
        self.region_of_interest = roi
        self.rise_duration = rise_duration
        self.fall_duration = fall_duration
        self.background_colour = background_color
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
    
    def get_layer_parameters(self) -> dict:
        return dict(self._layer_parameters)
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):

        # Update the faceMesh when switching between image and video processing
        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        # Masking does not support weight, so weight will always be 0.0 or 1.0
        weight = super().compute_weight(dt, self.supports_weight())

        # Occurs when the dt is less than the onset_time, or greater than the offset_time
        if weight == 0.0:
            return frame
        else:
            # Mask out the region of interest
            face_mesh = super().get_face_mesh()
            mask = mask_from_path(frame, self.region_of_interest, face_mesh)

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

            # Remove unwanted background inclusions in the masked area
            masked_frame = cv.bitwise_and(mask, foreground)
            masked_frame = np.reshape(masked_frame, (masked_frame.shape[0], masked_frame.shape[1], 1))
            masked_frame = np.where(masked_frame == 255, frame, self.background_colour)
            masked_frame = masked_frame.astype(np.uint8)
            return masked_frame

def layer_mask(time_onset:float=None, time_offset:float=None, timing_func:Callable[...,float]=timing_linear, region_of_interest:list[list[tuple]] | list[tuple]= FACE_OVAL_PATH, 
               background_colour:tuple[int,int,int] = (0,0,0), rise_duration:int=500, fall_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerMask:
    
    return LayerMask(background_colour, time_onset, time_offset, timing_func, region_of_interest, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)