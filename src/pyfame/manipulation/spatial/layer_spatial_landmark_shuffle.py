from pyfame.util.util_constants import *
from pyfame.mesh import *
from pyfame.util.util_exceptions import *
from pyfame.layer import layer
import cv2 as cv
import mediapipe as mp
import numpy as np
from operator import itemgetter
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class layer_spatial_landmark_shuffle(layer):
    def __init__(self, face_mesh:mp.solutions.face_mesh.FaceMesh, rand_seed:int|None):
        self.face_mesh = face_mesh
        self.rand_seed = rand_seed
    
    def apply_layer(self, frame:cv.typing.MatLike, weight:float, roi:list[list[tuple]] = [FACE_OVAL_TIGHT_PATH]) -> cv.typing.MatLike:
        rng = None
        output_frame = None
        if self.rand_seed is not None:
            rng = np.random.default_rng(self.rand_seed)
        else:
            rng = np.random.default_rng()

        # Precomputing shuffled grid positions
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        fo_screen_coords = get_mesh_coordinates_from_path(frame_rgb, self.face_mesh, FACE_OVAL_TIGHT_PATH)
        fo_mask = get_mask_from_path(frame, roi, self.face_mesh)

        # Get x and y bounds of the face oval
        max_x = max(fo_screen_coords, key=itemgetter(0))[0]
        min_x = min(fo_screen_coords, key=itemgetter(0))[0]

        max_y = max(fo_screen_coords, key=itemgetter(1))[1]
        min_y = min(fo_screen_coords, key=itemgetter(1))[1]

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

        le_screen_coords = get_mesh_coordinates_from_path(frame_rgb, self.face_mesh, LEFT_EYE_PATH)
        re_screen_coords = get_mesh_coordinates_from_path(frame_rgb, self.face_mesh, RIGHT_EYE_PATH)
        nose_screen_coords = get_mesh_coordinates_from_path(frame_rgb, self.face_mesh, NOSE_PATH)
        lips_screen_coords = get_mesh_coordinates_from_path(frame_rgb, self.face_mesh, MOUTH_PATH)

        # Creating boolean masks of each landmark region
        le_mask = get_mask_from_path(frame, [LEFT_EYE_PATH], self.face_mesh).astype(bool)
        re_mask = get_mask_from_path(frame, [RIGHT_EYE_PATH], self.face_mesh).astype(bool)
        nose_mask = get_mask_from_path(frame, [NOSE_PATH], self.face_mesh).astype(bool)
        lip_mask = get_mask_from_path(frame, [MOUTH_PATH], self.face_mesh).astype(bool)
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

            # Cut out landmark region and store it
            lm = cv.bitwise_and(src1=frame, src2=frame, mask=im_mask)
            lms.append((lm, (cy,cx)))

            # Fill landmark holes with inpainting
            output_frame[mask] = 0
            output_frame = cv.inpaint(output_frame, im_mask, 10, cv.INPAINT_NS)

        landmarks = dict(map(lambda i,j: (i,j), [1,2,3,4], lms))

        for key in landmarks:
            # Get the landmark, and the center point of its position
            landmark, center = landmarks[key]
            cx, cy = center
            h,w = landmark.shape[:2]

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
            
            cx += x_disp

            # Create landmark mask
            lm_mask = np.zeros((landmark.shape[0], landmark.shape[1]), dtype=np.uint8)
            lm_mask = np.where(landmark != 0, 255, 0)
            lm_mask = lm_mask.astype(np.uint8)
            
            # Clone the landmark onto the original face in its new position
            output_frame = cv.seamlessClone(landmark, output_frame, lm_mask, (cx, cy), cv.NORMAL_CLONE)
        
        return output_frame