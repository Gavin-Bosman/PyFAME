from pyfame.util.util_constants import *
from pyfame.mesh import *
from pyfame.mesh.get_mesh_landmarks import *
from pyfame.util.util_general_utilities import compute_rot_angle
from pyfame.util.util_exceptions import *
from pyfame.layer import layer
import cv2 as cv
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class layer_occlusion_path(layer):
    def __init__(self, face_mesh:mp.solutions.face_mesh.FaceMesh, landmarks_to_occlude:list[list[tuple]], fill:int = OCCLUSION_FILL_BLACK):
        self.face_mesh = face_mesh
        self.landmarks = landmarks_to_occlude
        self.fill = fill
    
    def apply_layer(self, frame, weight, roi):
        mask = get_mask_from_path(frame, self.landmarks, self.face_mesh)
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

        match self.fill:
            case 8:
                occluded = np.where(mask == 255, self.fill, frame)
                return occluded
            
            case 9:
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                fo_coords = get_mesh_coordinates_from_path(frame_rgb, self.face_mesh, FACE_OVAL_TIGHT_PATH)

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

class layer_occlusion_bar(layer):
    def __init__(self, face_mesh:mp.solutions.face_mesh.FaceMesh, landmarks_to_occlude:list[list[tuple]], color:tuple[int,int,int] = (0,0,0)):
        self.face_mesh = face_mesh
        self.landmarks = landmarks_to_occlude
        self.color = color
        self.min_x_lm_id = -1
        self.max_x_lm_id = -1
        self.compatible_paths = [LEFT_EYE_PATH, RIGHT_EYE_PATH, BOTH_EYES_PATH, NOSE_PATH, LIPS_PATH, MOUTH_PATH]
    
    def apply_layer(self, frame, weight, roi):
        lm_coords = get_mesh_coordinates(cv.cvtColor(frame, cv.COLOR_BGR2RGB), self.face_mesh)
        masked_frame = np.zeros_like(frame, dtype=np.uint8)
        refactored_lms = []

        for lm in self.landmarks:
            if lm not in self.compatible_paths:
                raise ValueError("Function has encountered an incompatible landmark path in layer_occlusion_bar.")
            elif lm == BOTH_EYES_PATH:
                refactored_lms.append(LEFT_EYE_PATH)
                refactored_lms.append(RIGHT_EYE_PATH)
            else:
                refactored_lms.append(lm)
        
        if self.landmarks != refactored_lms:
            self.landmarks = refactored_lms

        for lm in self.landmarks:

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