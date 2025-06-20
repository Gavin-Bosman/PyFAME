from pyfame.util.util_constants import *
from pyfame.mesh import get_mask_from_path
from pyfame.mesh.get_mesh_landmarks import *
from pyfame.layer import Layer
from pyfame.util.util_exceptions import *
from pyfame.timing.timing_curves import timing_constant
import cv2 as cv
import mediapipe as mp
import numpy as np
import logging
from typing import Callable

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class layer_color_brightness(Layer):
    def __init__(self, face_mesh:mp.solutions.face_mesh.FaceMesh, magnitude:float = 20.0, 
                 timing_func:Callable[...,float] = timing_constant, **timing_kwargs):
        self.magnitude = magnitude
        self.face_mesh = face_mesh
        self.timing_func = timing_func
        self.timing_kwargs = timing_kwargs
    
    def __get_timing_weight(self, dt:float) -> float:
        return self.timing_func(dt, **self.timing_kwargs)
    
    def apply_layer(self, frame:cv.typing.MatLike, roi:list[list[tuple]], dt:float):
        roi_mask = get_mask_from_path(frame, roi, self.face_mesh)
        weight = self.__get_timing_weight(dt)

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

        img_brightened = np.where(roi_mask == 255, cv.convertScaleAbs(src=frame, alpha=1, beta=(weight * self.magnitude)), frame)
        img_brightened[foreground == 0] = frame[foreground == 0]
        return img_brightened