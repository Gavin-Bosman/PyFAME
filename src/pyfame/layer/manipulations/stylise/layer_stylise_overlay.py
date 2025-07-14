from pyfame.mesh import *
from pyfame.file_access import *
from pyfame.utilities import compute_rot_angle
from pyfame.utilities.util_checks import *
from pyfame.utilities.util_constants import OVERLAY_SUNGLASSES, OVERLAY_GLASSES, OVERLAY_SWEAT, OVERLAY_TEAR
from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerStyliseOverlay(Layer):
    def __init__(self, overlay_type:int|str = "sunglasses", onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, 
                 roi:list[list[tuple]] = [FACE_OVAL_PATH], fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        super().__init__(onset_t, offset_t, timing_func, roi, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        check_type(overlay_type, [str, int])
        check_value(overlay_type, ["sunglasses", "glasses", "tear", "sweat", OVERLAY_SUNGLASSES, OVERLAY_GLASSES, OVERLAY_SWEAT, OVERLAY_TEAR])
        
        self.overlay_type = overlay_type
        self.previous_slope = None
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
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            landmark_screen_coords = get_mesh_coordinates(frame_rgb, face_mesh)
            face_oval_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, FACE_OVAL_PATH)
            overlayed_frame = frame.copy()

            # Get the facial width to scale the overlayed object
            max_x = max(x for x,_ in face_oval_coords)
            min_x = min(x for x,_ in face_oval_coords)

            match self.overlay_type:
                case 43 | "sunglasses":
                    # Read in sunglasses image
                    sunglasses = get_io_imread(".//overlay_images//sunglasses.png")

                    # Rescaling the overlay image to match facial width
                    overlay_width = sunglasses.shape[1]
                    overlay_height = sunglasses.shape[0]
                    scaling_factor = 1/(overlay_width/(max_x-min_x))
                    sunglasses = cv.resize(sunglasses, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)
                    overlay_width = sunglasses.shape[1]
                    overlay_height = sunglasses.shape[0]

                    rot_angle = 0.0
                    angle_to_x = 0.0

                    # Rotating the overlay 
                    # lm's 227 and 447 are used as the bounds of the sunglasses

                    if self.previous_slope is None:
                        p1 = landmark_screen_coords[227]
                        p2 = landmark_screen_coords[447]
                        cur_slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))
                        rot_angle = compute_rot_angle(slope_1=cur_slope)
                        self.previous_slope = cur_slope
                    else:
                        p1 = landmark_screen_coords[227]
                        p2 = landmark_screen_coords[447]
                        cur_slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))
                        rot_angle = compute_rot_angle(slope_1=cur_slope, slope_2=self.previous_slope )
                        angle_to_x = compute_rot_angle(slope_1=self.previous_slope)
                        self.previous_slope = cur_slope

                    # Add transparent padding prior to rotation
                    diag_size = int(np.ceil(np.sqrt(overlay_height**2 + overlay_width**2)))
                    pad_h = (diag_size-overlay_height)//2
                    pad_w = (diag_size-overlay_width)//2
                    padded = np.zeros((diag_size, diag_size, 4), dtype=np.uint8)
                    padded[pad_h:pad_h+overlay_height, pad_w:pad_w + overlay_width] = sunglasses

                    # Get center point of padded overlay
                    padded_height = padded.shape[0]
                    padded_width = padded.shape[1]
                    padded_center = (overlay_width//2, overlay_height//2)

                    # Perform rotation
                    if self.previous_slope is None:
                        rot_mat = cv.getRotationMatrix2D(padded_center, rot_angle, 1)
                    else:
                        rot_mat = cv.getRotationMatrix2D(padded_center, (rot_angle + angle_to_x), 1)
                        
                    sunglasses = cv.warpAffine(padded, rot_mat, (padded_width, padded_height), flags=cv.INTER_LINEAR)

                    overlay_img = sunglasses[:,:,:3]
                    overlay_mask = sunglasses[:,:,3] / 255.0
                    overlay_mask = overlay_mask[:,:,np.newaxis]
                    overlay_width = sunglasses.shape[1]
                    overlay_height = sunglasses.shape[0]

                    facial_center = landmark_screen_coords[6]
                    x_pos = facial_center.get('x') - padded_width//2
                    y_pos = facial_center.get('y') - padded_height//2

                    roi = frame[y_pos:y_pos + padded_height, x_pos:x_pos + padded_width]
                    blended = (1.0 - overlay_mask) * roi + overlay_mask * overlay_img

                    overlayed_frame[y_pos:y_pos + padded_height, x_pos:x_pos + padded_width] = blended.astype(np.uint8)
            
            return overlayed_frame
        
def layer_stylise_overlay(overlay_type:int|str, time_onset:float=None, time_offset:float=None, timing_function:Callable[...,float]=timing_linear, 
                 region_of_interest:list[list[tuple]] = [FACE_OVAL_PATH], fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):

    return LayerStyliseOverlay(overlay_type, time_onset, time_offset, timing_function, region_of_interest, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)