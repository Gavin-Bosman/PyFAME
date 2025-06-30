from pyfame.util.util_constants import *
from pyfame.mesh import get_mask_from_path
from pyfame.mesh.get_mesh_landmarks import FACE_OVAL_PATH
from pyfame.layer import Layer
from pyfame.util.util_checks import *
from pyfame.timing.timing_curves import timing_linear
import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerColorBrightness(Layer):
    def __init__(self, onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, roi:list[list[tuple]] = [FACE_OVAL_PATH], 
                 fade_duration:int = 500, magnitude:float = 20.0, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        super().__init__(onset_t, offset_t, timing_func, roi, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)

        check_type(magnitude, [float])

        self.roi = roi
        self.magnitude = magnitude
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.static_image_mode = False
    
    def supports_weight(self):
        return True

    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):
        weight = None

        if static_image_mode != self.static_image_mode:
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

            img_brightened = np.where(mask == 255, cv.convertScaleAbs(src=frame, alpha=1, beta=(weight * self.magnitude)), frame)
            img_brightened[foreground == 0] = frame[foreground == 0]
            return img_brightened