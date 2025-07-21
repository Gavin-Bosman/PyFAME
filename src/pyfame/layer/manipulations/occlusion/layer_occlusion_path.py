from pyfame.utilities.constants import *
from pyfame.mesh import *
from pyfame.utilities.checks import *
from pyfame.utilities.general_utilities import sanitize_json_value, get_roi_name
from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
from pyfame.layer.manipulations.mask import mask_from_path
import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerOcclusionPath(Layer):
    def __init__(self, fill_method:int|str = OCCLUSION_FILL_BLACK, onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, 
                 roi:list[list[tuple]] | list[tuple] = FACE_OVAL_PATH, rise_duration:int=500, fall_duration:int=500, min_tracking_confidence:float=0.5, min_detection_confidence:float=0.5, **kwargs):
        # Initialise superclass
        super().__init__(onset_t, offset_t, timing_func, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        self.static_image_mode = False
        self._pre_attrs = []
        self._pre_attrs = set(self.__dict__) # snapshot of just the superclass parameters
        
        # Perform input parameter checks
        check_type(fill_method, [int,str])
        check_value(fill_method, expected_values=[8,9,"black","mean"])

        # Declaring class parameters
        self.fill_method = fill_method
        self.time_onset = onset_t
        self.time_offset = offset_t
        self.timing_function = timing_func
        self.region_of_interest = roi
        self.rise_duration_msec = rise_duration
        self.fall_duration_msec = fall_duration
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
    
    def get_layer_parameters(self):
        return dict(self._layer_parameters)
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):
        
        # Update the faceMesh when switching between image and video processing
        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        weight = super().compute_weight(dt, self.supports_weight())
        
        if weight == 0.0:
            return frame
        else:
            # Mask out the region of interest
            face_mesh = super().get_face_mesh()
            mask = mask_from_path(frame, self.region_of_interest, face_mesh)
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

            match self.fill_method:
                case 8 | "black":
                    occluded = np.where(mask == 255, self.fill_method, frame)
                    return occluded
                
                case 9 | "mean":
                    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    # Uses the tight-path to avoid any background inclusion in the mean colour sampling
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
                         region_of_interest:list[list[tuple]] | list[tuple]=FACE_OVAL_PATH, rise_duration:int=500, fall_duration:int=500, min_tracking_confidence:float=0.5, 
                         min_detection_confidence:float=0.5, **kwargs) -> LayerOcclusionPath:
    
    return LayerOcclusionPath(fill_method, time_onset, time_offset, timing_function, region_of_interest, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)