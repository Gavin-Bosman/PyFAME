from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
from pyfame.mesh import *
from pyfame.utilities.general_utilities import get_roi_name, sanitize_json_value
from pyfame.utilities.checks import *
from pyfame.utilities.exceptions import *
from typing import Callable
import cv2 as cv
import numpy as np

class LayerStyliseOpticalFlowDense(Layer):
    def __init__(self, onset_t = None, offset_t = None, timing_func:Callable[...,float] = timing_linear, rise_duration = 500, 
                 fall_duration = 500, pixel_neighborhood_size:int = 5, search_window_size:int = 15, max_pyramid_level:int = 2, pyramid_scale:float = 0.5, 
                 max_iterations:int = 10, gaussian_deviation:float = 1.2, min_tracking_confidence = 0.5, min_detection_confidence = 0.5, **timing_kwargs):
        super().__init__(onset_t, offset_t, timing_func, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **timing_kwargs)

        self.loop_counter = 1
        self.hsv_mask = None
        self.previous_grey_frame = None
        self._pre_attrs = []
        self._pre_attrs = set(self.__dict__) # snapshot of just the superclass parameters

        # Perform parameter checks
        check_type(pixel_neighborhood_size, [int])
        check_type(search_window_size, [int])
        check_type(max_pyramid_level, [int])
        check_type(pyramid_scale, [float])
        check_value(pyramid_scale, min=0.0, max=1.0)
        check_type(max_iterations, [int])
        check_type(gaussian_deviation, [float])

        # Declare class parameters
        self.time_onset = onset_t
        self.time_offset = offset_t
        self.timing_function = timing_func
        self.rise_duration = rise_duration
        self.fall_duration = fall_duration
        self.pixel_neighborhood_size = pixel_neighborhood_size
        self.search_window_size = search_window_size
        self.max_pyramid_level = max_pyramid_level
        self.pyramid_scale = pyramid_scale
        self.max_iterations = max_iterations
        self.gaussian_deviation = gaussian_deviation
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.timing_kwargs = timing_kwargs

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
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float, static_image_mode:bool = False):
        # Update the faceMesh when switching between image and video processing
        if static_image_mode == True:
            raise UnrecognizedExtensionError(message="Dense optical flow does not support static image files.")

        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            if self.loop_counter == 1:
                self.hsv_mask = np.zeros_like(frame)
                self.previous_grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                self.hsv_mask[...,1] = 255

                return frame

            if self.loop_counter > 1:
                grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # Calculate dense optical flow
                flow = cv.calcOpticalFlowFarneback(self.previous_grey_frame, grey_frame, None, self.pyramid_scale, self.max_pyramid_level, 
                                                   self.search_window_size, self.max_iterations, self.pixel_neighborhood_size, self.gaussian_deviation, 0)

                # Get vector magnitudes and angles
                magnitudes, angles = cv.cartToPolar(flow[...,0],flow[...,1])

                self.hsv_mask[...,0] = angles * (180/(np.pi/2))
                self.hsv_mask[...,2] = cv.normalize(magnitudes, None, 0, 255, cv.NORM_MINMAX)

                output_img = cv.cvtColor(self.hsv_mask, cv.COLOR_HSV2BGR)

                self.previous_grey_frame = grey_frame.copy()

                return output_img

def layer_stylise_optical_flow_dense(onset_t = None, offset_t = None, timing_function:Callable[...,float] = timing_linear, rise_duration = 500, 
                                     fall_duration = 500, pixel_neighborhood_size:int = 5, search_window_size:int = 15, max_pyramid_level:int = 2, 
                                     pyramid_scale:float = 0.5, max_iterations:int = 10, gaussian_deviation:float = 1.2, min_tracking_confidence = 0.5, 
                                     min_detection_confidence = 0.5, **timing_kwargs) -> LayerStyliseOpticalFlowDense:
    
    return LayerStyliseOpticalFlowDense(onset_t, offset_t, timing_function, rise_duration, fall_duration, pixel_neighborhood_size, search_window_size, 
                                        max_pyramid_level, pyramid_scale, max_iterations, gaussian_deviation, min_tracking_confidence, min_detection_confidence, **timing_kwargs)