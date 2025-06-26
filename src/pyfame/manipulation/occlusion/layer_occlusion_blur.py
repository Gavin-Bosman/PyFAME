from pyfame.util.util_constants import *
from pyfame.mesh import *
from pyfame.util.util_checks import *
from pyfame.layer import Layer
import cv2 as cv
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerOcclusionBlur(Layer):
    def __init__(self, method:str|int = "gaussian", kernel_size:int = 15):
        check_type(method, [str, int])
        check_value(method, [11,12,13,"average","gaussian","median"])
        check_type(kernel_size, [int])
        check_value(kernel_size, min=1)

        if isinstance(method, str):
            self.method = str.lower(method)
        else:
            self.method = method

        self.k_size = kernel_size
    
    def supports_weight(self):
        return False
    
    def apply_layer(self, face_mesh:mp.solutions.face_mesh.FaceMesh, frame, roi, weight):

        if weight == 0.0:
            return frame
        else:
            mask = get_mask_from_path(frame, roi, face_mesh)
            output_frame = np.zeros_like(frame, dtype=np.uint8)

            match self.method:
                case "average" | 11:
                    frame_blurred = cv.blur(frame, (self.k_size, self.k_size))
                    output_frame = np.where(mask == 255, frame_blurred, frame)
                
                case "gaussian" | 12:
                    frame_blurred = cv.GaussianBlur(frame, (self.k_size, self.k_size), 0)
                    output_frame = np.where(mask == 255, frame_blurred, frame)
                
                case "median" | 13:
                    frame_blurred = cv.medianBlur(frame, self.k_size)
                    output_frame = np.where(mask == 255, frame_blurred, frame)
            
            return output_frame