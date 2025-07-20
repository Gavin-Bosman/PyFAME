from pyfame.utilities.constants import *
from pyfame.mesh import *
from pyfame.utilities.checks import *
from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
from pyfame.layer.manipulations.mask import mask_from_path
import cv2 as cv
import numpy as np
from operator import itemgetter
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerSpatialLandmarkRelocate(Layer):
    def __init__(self, rand_seed:int|None, onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, 
                 roi:list[list[tuple]] | list[tuple]=FACE_OVAL_PATH, rise_duration:int=500, fade_duration:int=500, min_tracking_confidence:float=0.5, min_detection_confidence:float=0.5, **kwargs):
        # Initialise the superclass
        super().__init__(onset_t, offset_t, timing_func, rise_duration, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        # Perform input parameter checks
        check_type(rand_seed, [int, type(None)])

        # Declare class parameters
        self.rand_seed = rand_seed
        self.roi = roi
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.static_image_mode = False
    
    def supports_weight(self):
        return False
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float, static_image_mode:bool = False) -> cv.typing.MatLike:
        # Update the faceMesh when switching between image and video processing
        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            # Create an rng instance 
            rng = None
            if self.rand_seed is not None:
                rng = np.random.default_rng(self.rand_seed)
            else:
                rng = np.random.default_rng()

            # Get the face oval coordinates and mask out the roi
            face_mesh = super().get_face_mesh()
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            fo_screen_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, FACE_OVAL_TIGHT_PATH)
            fo_mask = mask_from_path(frame, self.roi, self.face_mesh)

            # Get x and y bounds of the face oval
            max_x = max(fo_screen_coords, key=itemgetter(0))[0]
            min_x = min(fo_screen_coords, key=itemgetter(0))[0]

            max_y = max(fo_screen_coords, key=itemgetter(1))[1]
            min_y = min(fo_screen_coords, key=itemgetter(1))[1]

            # The angle of rotation and x,y positons of the landmarks are randomly generated in the loop below
            rot_angles = {}
            x_displacements = {}

            for i in range(4):
                rn = rng.random()

                if i+1 < 3:
                        if rn < 0.25:
                            rot_angles.update({i+1:90})
                        elif rn < 0.5:
                            rot_angles.update({i+1:-90})
                        elif rn < 0.75:
                            rot_angles.update({i+1:180})
                        else:
                            rot_angles.update({i+1:0})
                elif i+1 == 3:
                    if rn < 0.5:
                        rot_angles.update({i+1:90})
                    else:
                        rot_angles.update({i+1:-90})
                else:
                    if rn < 0.5:
                        rot_angles.update({i+1:180})
                    else:
                        rot_angles.update({i+1:0})
                
                if rn < 0.5:
                    x_displacements.update({i+1:int(-40 * rng.random())})
                else:
                    x_displacements.update({i+1:int(40 * rng.random())})

            # Get the pixel coordinates of various landmark regions
            le_screen_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, LEFT_EYE_PATH)
            re_screen_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, RIGHT_EYE_PATH)
            nose_screen_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, NOSE_PATH)
            lips_screen_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, MOUTH_PATH)

            # Creating boolean masks of each landmark region
            le_mask = mask_from_path(frame, [LEFT_EYE_PATH], face_mesh).astype(bool)
            re_mask = mask_from_path(frame, [RIGHT_EYE_PATH], face_mesh).astype(bool)
            nose_mask = mask_from_path(frame, [NOSE_PATH], face_mesh).astype(bool)
            lip_mask = mask_from_path(frame, [MOUTH_PATH], face_mesh).astype(bool)
            fo_mask = fo_mask.astype(bool)

            masks = [le_mask, re_mask, nose_mask, lip_mask]
            screen_coords = [le_screen_coords, re_screen_coords, nose_screen_coords, lips_screen_coords]
            lms = []
            output_frame = frame.copy()

            # Cut out, and store landmarks
            for mask, coords in zip(masks, screen_coords):
                im_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                im_mask[mask] = 255

                max_x = max(coords, key=itemgetter(0))[0]
                min_x = min(coords, key=itemgetter(0))[0]

                max_y = max(coords, key=itemgetter(1))[1]
                min_y = min(coords, key=itemgetter(1))[1]

                # Compute the center bisecting lines of the landmark
                cx = round((max_y + min_y)/2)           
                cy = round((max_x + min_x)/2)

                # Cut out the current landmark region and store it
                lm = cv.bitwise_and(src1=frame, src2=frame, mask=im_mask)
                lms.append((lm, (cy,cx)))

                # Fill landmark holes with navier-stokes inpainting;
                # uses nearest-neighbor colour sampling to fill in gaps in an image
                output_frame[mask] = 0
                output_frame = cv.inpaint(output_frame, im_mask, 10, cv.INPAINT_NS)

            landmarks = dict(map(lambda i,j: (i,j), [1,2,3,4], lms))

            for key in landmarks:
                # Get the landmark, and the center point of its position
                landmark, center = landmarks[key]
                cx, cy = center
                h,w = landmark.shape[:2]

                # Get the current landmarks randomly generated rotation angle and x displacement
                rot_angle = rot_angles[key]
                x_disp = x_displacements[key]

                # Generate rotation matrices for the landmark
                if key == 3:
                    rot_mat = cv.getRotationMatrix2D(center=center, angle=rot_angle, scale=1)
                    landmark = cv.warpAffine(landmark, rot_mat, (w,h))
                    cy += 20
                else:
                    rot_mat = cv.getRotationMatrix2D(center=center, angle=rot_angle, scale=1)
                    landmark = cv.warpAffine(landmark, rot_mat, (w,h))
                
                # Add the x displacement to the original center point to translate it
                cx += x_disp

                # Create landmark mask
                lm_mask = np.zeros((landmark.shape[0], landmark.shape[1]), dtype=np.uint8)
                lm_mask = np.where(landmark != 0, 255, 0)
                lm_mask = lm_mask.astype(np.uint8)
                
                # Clone the landmark onto the original face in its new position
                output_frame = cv.seamlessClone(landmark, output_frame, lm_mask, (cx, cy), cv.NORMAL_CLONE)
            
            return output_frame
        
def layer_spatial_landmark_relocate(rand_seed:int|None, time_onset:float=None, time_offset:float=None, timing_function:Callable[...,float]=timing_linear, 
                 region_of_interest:list[list[tuple]] | list[tuple] = FACE_OVAL_PATH, rise_duration:int=500, fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerSpatialLandmarkRelocate:
    
    return LayerSpatialLandmarkRelocate(rand_seed, time_onset, time_offset, timing_function, region_of_interest, rise_duration, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)