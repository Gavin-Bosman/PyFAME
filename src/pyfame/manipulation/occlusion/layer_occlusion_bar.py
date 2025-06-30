from pyfame.util.util_constants import *
from pyfame.mesh import *
from pyfame.util.util_general_utilities import compute_rot_angle
from pyfame.util.util_checks import *
from pyfame.layer import Layer
from pyfame.timing.timing_curves import timing_linear
import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerOcclusionBar(Layer):
    def __init__(self, bar_color:tuple[int,int,int] = (0,0,0), onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, 
                 roi:list[list[tuple]] = [FACE_OVAL_PATH], fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        super().__init__(onset_t, offset_t, timing_func, roi, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        check_type(bar_color, [tuple])
        check_type(bar_color, [int], iterable=True)
        for i in bar_color:
            check_value(i, min=0, max=255)

        self.color = bar_color
        self.min_x_lm_id = -1
        self.max_x_lm_id = -1
        self.compatible_paths = [LEFT_EYE_PATH, RIGHT_EYE_PATH, BOTH_EYES_PATH, NOSE_PATH, LIPS_PATH, MOUTH_PATH]
        self.roi = roi
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.static_image_mode = False
    
    def supports_weight(self):
        return False
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float, static_image_mode:bool = False):

        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            face_mesh = super().get_face_mesh()
            lm_coords = get_mesh_coordinates(cv.cvtColor(frame, cv.COLOR_BGR2RGB), face_mesh)
            masked_frame = np.zeros_like(frame, dtype=np.uint8)
            refactored_lms = []

            for lm in self.roi:
                if lm not in self.compatible_paths:
                    raise ValueError("Function has encountered an incompatible landmark path in layer_occlusion_bar.")
                elif lm == BOTH_EYES_PATH:
                    refactored_lms.append(LEFT_EYE_PATH)
                    refactored_lms.append(RIGHT_EYE_PATH)
                else:
                    refactored_lms.append(lm)

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
            rot_angle = compute_rot_angle(slope1=slope)

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
            
            output_frame = np.where(masked_frame == 255, self.color, frame)
            return output_frame.astype(np.uint8)