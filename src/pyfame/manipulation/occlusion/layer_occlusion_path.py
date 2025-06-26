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

class LayerOcclusionPath(Layer):
    def __init__(self, fill_method:int|str = OCCLUSION_FILL_BLACK):
        check_type(fill_method, [int,str])
        check_value(fill_method, expected_values=[8,9,"black","mean"])

        if isinstance(fill_method, str):
            self.fill = str.lower(fill_method)
        else:
            self.fill = fill_method
    
    def supports_weight(self):
        return False
    
    def apply_layer(self, face_mesh:mp.solutions.face_mesh.FaceMesh, frame:cv.typing.MatLike, roi:list[list[tuple]], weight:float):

        if weight == 0.0:
            return frame
        else:
            mask = get_mask_from_path(frame, roi, face_mesh)
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

            match self.fill:
                case 8 | "black":
                    occluded = np.where(mask == 255, self.fill, frame)
                    return occluded
                
                case 9 | "mean":
                    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    fo_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, FACE_OVAL_TIGHT_PATH)

                    # Creating boolean masks for the facial landmarks 
                    bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    bool_mask = cv.fillConvexPoly(bool_mask, np.array(fo_coords), 1)
                    bool_mask = bool_mask.astype(bool)

                    # Extracting the mean pixel value of the face
                    bin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    bin_mask[bool_mask] = 255

                    mean = cv.mean(frame, bin_mask)

                    # Fill occlusion regions with facial mean
                    mean_img = np.zeros_like(frame, dtype=np.uint8)
                    mean_img[:] = mean[:3]
                    occluded = np.where(mask == 255, mean_img, frame)
                    return occluded

