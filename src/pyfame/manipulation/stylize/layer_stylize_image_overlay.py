from pyfame.mesh import *
from pyfame.io import *
from pyfame.util import compute_rot_angle
from pyfame.util.util_checks import *
from pyfame.util.util_constants import OVERLAY_SUNGLASSES, OVERLAY_GLASSES, OVERLAY_SWEAT, OVERLAY_TEAR
from pyfame.layer import Layer
import cv2 as cv
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerStylizeOverlay(Layer):
    def __init__(self, overlay_type:int|str):
        check_type(overlay_type, [str, int])
        check_value(overlay_type, ["sunglasses", "glasses", "tear", "sweat", OVERLAY_SUNGLASSES, OVERLAY_GLASSES, OVERLAY_SWEAT, OVERLAY_TEAR])
        
        self.overlay_type = overlay_type
        self.previous_slope = None
    
    def supports_weight(self):
        return False
    
    def apply_layer(self, face_mesh:mp.solutions.face_mesh.FaceMesh, frame:cv.typing.MatLike, roi:list[list[tuple]], weight:float):
        if weight == 0.0:
            return frame
        else:
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
                        rot_angle = compute_rot_angle(slope1=cur_slope)
                        self.previous_slope = cur_slope
                    else:
                        p1 = landmark_screen_coords[227]
                        p2 = landmark_screen_coords[447]
                        cur_slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))
                        rot_angle = compute_rot_angle(slope1=cur_slope, slope2=self.previous_slope )
                        angle_to_x = compute_rot_angle(slope1=self.previous_slope)
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