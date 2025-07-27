from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
from pyfame.layer.manipulations.mask import mask_from_path
from pyfame.mesh import *
from pyfame.utilities.general_utilities import get_roi_name, sanitize_json_value
from pyfame.utilities.checks import *
from pyfame.utilities.exceptions import *
from typing import Callable
import cv2 as cv
import numpy as np

# TODO after creating dataclass, add vector line width param

class LayerStyliseOpticalFlowSparse(Layer):
    def __init__(self, time_onset:float = None, time_offset:float = None, rise_duration:float = 500, fall_duration:float = 500, 
                 timing_function:Callable[...,float] = timing_linear, landmarks_to_track:list[int] | None = None, 
                 max_corners:int = 20, corner_quality_level:float = 0.3, min_corner_distance:int = 7, block_size:tuple[int] = (5,5), 
                 search_window_size:tuple[int] = (15,15), max_pyramid_level:int = 2, pyramid_scale:float = 0.5, gaussian_deviation:float = 1.2, 
                 max_iterations:int = 10, accuracy_threshold:float = 0.03, point_colour:tuple[int] = (0,0,0), point_radius:int = 4, 
                 vector_colour:tuple[int] = (0,0,255), min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5, **timing_kwargs):
        # Initialise superclass
        super().__init__(time_onset, time_offset, timing_function, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **timing_kwargs)
        self.static_image_mode = False
        self.initial_points = []
        self.previous_grey_frame = None
        self.vector_mask = None
        self.loop_counter = 1
        self._pre_attrs = []
        self._pre_attrs = set(self.__dict__) # snapshot of just the superclass parameters

        # Perform parameter checks
        if landmarks_to_track is not None:
            check_type(landmarks_to_track, [int], iterable=True)
        check_type(max_corners, [int])
        check_type(corner_quality_level, [float])
        check_value(corner_quality_level, min=0.0, max=1.0)
        check_type(min_corner_distance, [int])
        check_type(block_size, [int], iterable=True)
        check_type(search_window_size, [int], iterable=True)
        check_type(max_pyramid_level, [int])
        check_type(pyramid_scale, [float])
        check_value(pyramid_scale, min=0.0, max=1.0)
        check_type(gaussian_deviation, [float])
        check_type(max_iterations, [int])
        check_type(accuracy_threshold, [float])
        check_value(accuracy_threshold, min=0.0, max=1.0)
        check_type(point_colour, [int], iterable=True)
        check_type(point_radius, [int])
        check_type(vector_colour, [int], iterable=True)

        # Declare class parameters
        self.time_onset = time_onset
        self.time_offset =  time_offset
        self.rise_duration = rise_duration
        self.fall_duration = fall_duration
        self.timing_function = timing_function
        self.landmark_idx_to_track = landmarks_to_track
        self.max_points = max_corners
        self.point_quality_threshold = corner_quality_level
        self.min_point_distance = min_corner_distance
        self.pixel_neighborhood_size = block_size
        self.search_window_size = search_window_size
        self.max_pyramid_level = max_pyramid_level
        self.pyramid_scale = pyramid_scale
        self.gaussian_deviation = gaussian_deviation
        self.max_iterations = max_iterations
        self.flow_accuracy_threshold = accuracy_threshold
        self.point_colour = point_colour
        self.point_radius = point_radius
        self.vector_colour = vector_colour
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
            raise UnrecognizedExtensionError(message="Sparse optical flow does not support static image files.")

        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            # Defining persistent loop params
            output_img = None

            # Parameters for lucas kanade optical flow
            lk_params = dict(winSize  = self.search_window_size,
                maxLevel = self.max_pyramid_level,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, self.max_iterations, self.flow_accuracy_threshold))

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            face_mesh = super().get_face_mesh()
            landmark_screen_coords = get_mesh_coordinates(frame_rgb, face_mesh)
            
            # Create face oval image mask
            face_mask = mask_from_path(frame, FACE_OVAL_PATH, face_mesh)

            # Main Processing loop
            
            if self.loop_counter == 1:
                self.vector_mask = np.zeros_like(frame)
                self.previous_grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # If landmarks were provided 
                if self.landmark_idx_to_track is not None:
                    self.initial_points = np.array([[lm.get('x'), lm.get('y')] for lm in landmark_screen_coords if lm.get('id') in self.landmark_idx_to_track], dtype=np.float32)
                    self.initial_points = self.initial_points.reshape(-1,1,2)
                else:
                    self.initial_points = cv.goodFeaturesToTrack(self.previous_grey_frame, self.max_points, self.point_quality_threshold, self.min_point_distance, self.pixel_neighborhood_size, mask=face_mask)
                
                self.loop_counter += 1
                return frame

            if self.loop_counter > 1:
                grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # Calculate optical flow
                cur_points, st, err = cv.calcOpticalFlowPyrLK(self.previous_grey_frame, grey_frame, self.initial_points, None, **lk_params)

                # Select good points
                good_new_points = None
                good_old_points = None
                if cur_points is not None:
                    good_new_points = cur_points[st==1]
                    good_old_points = self.initial_points[st==1]
                else:
                    return frame

                # Draw optical flow vectors and write out values
                for new, old in zip(good_new_points, good_old_points):
                    x0, y0 = old.ravel()
                    x1, y1 = new.ravel()

                    # Draw optical flow vectors on output frame
                    self.vector_mask = cv.line(self.vector_mask, (int(x0), int(y0)), (int(x1), int(y1)), self.vector_colour, 2)

                    frame = cv.circle(frame, (int(x1), int(y1)), self.point_radius, self.point_colour, -1)
                    
                output_img = cv.add(frame, self.vector_mask)

                # Update previous frame and points
                self.previous_grey_frame = grey_frame.copy()
                self.initial_points = good_new_points.reshape(-1, 1, 2)

                return output_img
                
def layer_stylise_optical_flow_sparse(time_onset:float = None, time_offset:float = None, rise_duration:float = 500, fall_duration:float = 500, 
                                      timing_function:Callable[...,float] = timing_linear, landmarks_to_track:list[int] | None = None, 
                                      max_points:int = 20, point_quality_threshold:float = 0.3, min_point_distance:int = 7, pixel_neighborhood_size:tuple[int] = (5,5), 
                                      search_window_size:tuple[int] = (15,15), max_pyramid_level:int = 2, pyramid_scale:float = 0.5, gaussian_deviation:float = 1.2, 
                                      max_iterations:int = 10, flow_accuracy_threshold:float = 0.03, point_colour:tuple[int] = (0,0,0), point_radius:int = 4, 
                                      vector_colour:tuple[int] = (0,0,255), min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5, **timing_kwargs) -> LayerStyliseOpticalFlowSparse:
    
    return LayerStyliseOpticalFlowSparse(time_onset, time_offset, rise_duration, fall_duration, timing_function, landmarks_to_track, max_points, point_quality_threshold, 
                                         min_point_distance, pixel_neighborhood_size, search_window_size, max_pyramid_level, pyramid_scale, gaussian_deviation, max_iterations, 
                                         flow_accuracy_threshold, point_colour, point_radius, vector_colour, min_detection_confidence, min_tracking_confidence, **timing_kwargs)