from pyfame.util.util_constants import *
from pyfame.util.util_exceptions import *
from pyfame.mesh import get_mask_from_path
from pyfame.layer import Layer
from pyfame.timing.timing_curves import timing_constant
import numpy as np
import cv2 as cv
import mediapipe as mp
import logging
from typing import Callable

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class layer_mask(Layer):
    def __init__(self, face_mesh:mp.solutions.face_mesh.FaceMesh, background_color:tuple[int,int,int] = (0,0,0)):
        self.face_mesh = face_mesh
        self.background_color = background_color
    
    def apply_layer(self, frame, roi, dt):
        mask = get_mask_from_path(frame, roi, self.face_mesh)

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