from pyfame.util.util_constants import *
from pyfame.mesh import get_mask_from_path
from pyfame.layer import Layer
from pyfame.util.util_exceptions import *
from pyfame.util.util_checks import *
import cv2 as cv
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerColor(Layer):

    def __init__(self, focus_color:str|int, magnitude:float = 10.0):
        
        check_type(focus_color, [str,int])
        check_value(focus_color, ["red", "green", "blue", "yellow", 4, 5, 6, 7])
        check_type(magnitude, [float])
        check_value(magnitude, min=0)

        self.magnitude = magnitude
        self.color = focus_color
    
    def supports_weight(self) -> bool:
        return True
    
    def apply_layer(self, face_mesh:mp.solutions.face_mesh.FaceMesh, frame:cv.typing.MatLike, roi:list[list[tuple]], weight:float):
        mask = get_mask_from_path(frame, roi, face_mesh)

        # Convert input image to CIE La*b* color space (perceptually uniform space)
        img_LAB = cv.cvtColor(frame, cv.COLOR_BGR2LAB).astype(np.float32)
        l,a,b = cv.split(img_LAB)

        match self.color:
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
        
        img_LAB = cv.merge([l,a,b])

        # Convert CIE La*b* back to BGR
        result = cv.cvtColor(img_LAB.astype(np.uint8), cv.COLOR_LAB2BGR)
        return result