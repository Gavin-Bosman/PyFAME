from pyfame.util.util_constants import *
from pyfame.mesh import *
from pyfame.util.util_exceptions import *
from pyfame.layer import layer
import cv2 as cv
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class layer_color_saturation(layer):
    def __init__(self, face_mesh:mp.solutions.face_mesh.FaceMesh, magnitude:float):
        self.face_mesh = face_mesh
        self.magnitude = magnitude
    
    def apply_layer(self, frame, weight, roi):
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

        img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)
        h,s,v = cv.split(img_hsv)
        s = np.where(mask == 255, s + (weight * self.magnitude), s)
        np.clip(s,0,255)
        img_hsv = cv.merge([h,s,v])

        img_bgr = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
        img_bgr[foreground == 0] = frame[foreground == 0]
        return img_bgr