from pyfame.utilities.util_constants import *
from pyfame.mesh import *
from pyfame.utilities.util_checks import *
from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerOcclusionPath(Layer):
    def __init__(self, fill_method:int|str = OCCLUSION_FILL_BLACK, onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, 
                 roi:list[list[tuple]] = [FACE_OVAL_PATH], fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        super().__init__(onset_t, offset_t, timing_func, roi, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        check_type(fill_method, [int,str])
        check_value(fill_method, expected_values=[8,9,"black","mean"])

        self.fill = fill_method
        self.roi = roi
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.static_image_mode = False
    
    def supports_weight(self):
        return False
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):
        
        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        weight = super().compute_weight(dt, self.supports_weight())
        face_mesh = super().get_face_mesh()

        if weight == 0.0:
            return frame
        else:
            mask = get_mask_from_path(frame, self.roi, face_mesh)
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

def layer_occlusion_path(fill_method:int|str = OCCLUSION_FILL_BLACK, time_onset:float=None, time_offset:float=None, timing_function:Callable[...,float]=timing_linear, 
                 region_of_interest:list[list[tuple]] = [FACE_OVAL_PATH], fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerOcclusionPath:
    
    return LayerOcclusionPath(fill_method, time_onset, time_offset, timing_function, region_of_interest, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)