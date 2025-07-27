from pyfame.utilities.constants import *
from pyfame.mesh import *
from pyfame.utilities.general_utilities import compute_rot_angle, sanitize_json_value, get_roi_name
from pyfame.utilities.checks import *
from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerOcclusionBar(Layer):
    def __init__(self, bar_color:tuple[int,int,int] = (0,0,0), onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, 
                 roi:list[list[tuple]] | list[tuple] = FACE_OVAL_PATH, rise_duration:int=500, fall_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        # Initialise superclass
        super().__init__(onset_t, offset_t, timing_func, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        self.min_x_lm_id = -1
        self.max_x_lm_id = -1
        self.static_image_mode = False
        self.compatible_paths = [LEFT_EYE_PATH, RIGHT_EYE_PATH, BOTH_EYES_PATH, NOSE_PATH, LIPS_PATH, MOUTH_PATH]
        self._pre_attrs = []
        self._pre_attrs = set(self.__dict__) # snapshot of just the superclass parameters
        
        # Perform parameter checks
        check_type(bar_color, [tuple])
        check_type(bar_color, [int], iterable=True)
        for i in bar_color:
            check_value(i, min=0, max=255)

        # Define class parameters
        self.bar_color = bar_color
        self.time_onset = onset_t
        self.time_offset = offset_t
        self.timing_function = timing_func
        self.region_of_interest = roi
        self.rise_duration_msec = rise_duration
        self.fall_duration_msec = fall_duration
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.timing_kwargs = kwargs

        # Check for incompatible landmark paths, handle error cases
        if isinstance(roi[0], list):
            for lm in roi:
                if lm not in self.compatible_paths:
                    print("An incompatible landmark path has been provided to LayerOcclusionBar")
                    raise ValueError()
        else:
            if roi not in self.compatible_paths:
                print("An incompatible landmark path has been provided to LayerOcclusionBar")
                raise ValueError()
        
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

        # Update faceMesh when switching between image and video processing
        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        # Bar occlusion does not support weight, so weight will always be 0.0 or 1.0
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            # Get the faceMesh coordinate set
            face_mesh = super().get_face_mesh()
            lm_coords = get_mesh_coordinates(cv.cvtColor(frame, cv.COLOR_BGR2RGB), face_mesh)
            masked_frame = np.zeros_like(frame, dtype=np.uint8)
            refactored_lms = []
            
            # Replace placeholder concave path with its convex sub-paths
            if isinstance(self.region_of_interest[0], list):
                for lm in self.region_of_interest:
                    if lm == BOTH_EYES_PATH:
                        refactored_lms.append(LEFT_EYE_PATH)
                        refactored_lms.append(RIGHT_EYE_PATH)
                    else:
                        refactored_lms.append(lm)
            elif self.region_of_interest == BOTH_EYES_PATH:
                refactored_lms.append(LEFT_EYE_PATH)
                refactored_lms.append(RIGHT_EYE_PATH)

            for lm in refactored_lms:

                min_x = 1000
                max_x = 0

                # find the two points closest to the beginning and end x-positions of the landmark region
                unique_landmarks = np.unique(lm)
                for lm_id in unique_landmarks:
                    cur_lm = lm_coords[lm_id]
                    if cur_lm.get('x') < min_x:
                        min_x = cur_lm.get('x')
                        self.min_x_lm_id = lm_id
                    if cur_lm.get('x') > max_x:
                        max_x = cur_lm.get('x')
                        self.max_x_lm_id = lm_id

            # Calculate the slope of the connecting line & angle to the horizontal
            p1 = lm_coords[self.min_x_lm_id]
            p2 = lm_coords[self.max_x_lm_id]
            slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))
            rot_angle = compute_rot_angle(slope_1=slope)

            # Compute the center bisecting line of the landmark
            cx = round((p2.get('y') + p1.get('y'))/2)
            cy = round((p2.get('x') + p1.get('x'))/2)
            
            # Generate the rectangle
            rectangle = cv.rectangle(masked_frame, (p1.get('x')-50, cx - 50), (p2.get('x') + 50, cx + 50), (255,255,255), -1)
            masked_frame_t = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
            
            # Generate rotation matrix and rotate the rectangle
            rot_mat = cv.getRotationMatrix2D((cy,cx), (rot_angle), 1)
            rot_img = cv.warpAffine(rectangle, rot_mat, (masked_frame_t.shape[1], masked_frame_t.shape[0]))
            
            masked_frame = cv.bitwise_or(masked_frame, np.where(rot_img == 255, 255, masked_frame_t))
            
            output_frame = np.where(masked_frame == 255, self.bar_color, frame)
            return output_frame.astype(np.uint8)

def layer_occlusion_bar(bar_color:tuple[int,int,int] = (0,0,0), time_onset:float=None, time_offset:float=None, timing_function:Callable[...,float]=timing_linear, 
                 region_of_interest:list[list[tuple]] | list[tuple] = FACE_OVAL_PATH, rise_duration:int=500, fall_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerOcclusionBar:
    
    return LayerOcclusionBar(bar_color, time_onset, time_offset, timing_function, region_of_interest, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)