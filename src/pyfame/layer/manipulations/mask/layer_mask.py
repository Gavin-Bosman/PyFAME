from pyfame.utilities.util_constants import *
from pyfame.utilities.util_checks import *
from pyfame.mesh import get_mask_from_path
from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
from pyfame.mesh.get_mesh_landmarks import FACE_OVAL_PATH
import numpy as np
import cv2 as cv
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerMask(Layer):
    def __init__(self, background_color:tuple[int,int,int] = (0,0,0), onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, 
                 roi:list[list[tuple]] = [FACE_OVAL_PATH], fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        super().__init__(onset_t, offset_t, timing_func, roi, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        check_type(background_color, [tuple])
        check_type(background_color, [int], iterable=True)
        for i in background_color:
            check_value(i, min=0, max=255)

        self.roi = roi
        self.background_color = background_color
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.static_image_mode = False
    
    def supports_weight(self):
        return False
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):
        weight = None

        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            face_mesh = super().get_face_mesh()
            mask = get_mask_from_path(frame, self.roi, face_mesh)

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
            masked_frame = np.where(masked_frame == 255, frame, self.background_color)
            masked_frame = masked_frame.astype(np.uint8)
            return masked_frame

def layer_mask(time_onset:float=None, time_offset:float=None, timing_func:Callable[...,float]=timing_linear, region_of_interest:list[list[tuple]] = [FACE_OVAL_PATH], 
               background_color:tuple[int,int,int] = (0,0,0), fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerMask:
    
    return LayerMask(background_color, time_onset, time_offset, timing_func, region_of_interest, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)