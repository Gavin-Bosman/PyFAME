from pyfame.utils.predefined_constants import *
from pyfame.mesh.landmarks import *
from pyfame.utils.utils import get_variable_name, compute_rot_angle
from pyfame.utils.exceptions import *
from pyfame.io import *
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
from skimage.util import *
from operator import itemgetter
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def get_mask_from_path(frame:cv.typing.MatLike, roi_path:list[tuple], face_mesh:mp.solutions.face_mesh.FaceMesh) -> cv.typing.MatLike:
    
    face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    landmark_screen_coords = []
    masked_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:

            # Convert normalised landmark coordinates to x-y pixel coordinates
            for id,lm in enumerate(face_landmarks.landmark):
                ih, iw, ic = frame.shape
                x,y = int(lm.x * iw), int(lm.y * ih)
                landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
    else:
        debug_logger.error("Function encountered a FaceNotFoundError attempting to call FaceMesh.process().")
        logger.error("Function encountered a FaceNotFoundError attempting to initialize the mediapipe face landmarker.")
        raise FaceNotFoundError()

    match roi_path:
        # Both Cheeks
        case [(0,)]:
            lc_screen_coords = []
            rc_screen_coords = []

            left_cheek_path = create_path(LEFT_CHEEK_IDX)
            right_cheek_path = create_path(RIGHT_CHEEK_IDX)

            # Left cheek screen coordinates
            for cur_source, cur_target in left_cheek_path:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                lc_screen_coords.append((source.get('x'),source.get('y')))
                lc_screen_coords.append((target.get('x'),target.get('y')))
            
            # Right cheek screen coordinates
            for cur_source, cur_target in right_cheek_path:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                rc_screen_coords.append((source.get('x'),source.get('y')))
                rc_screen_coords.append((target.get('x'),target.get('y')))
            
            lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
            lc_screen_coords.reshape((-1, 1, 2))

            lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
            lc_mask = lc_mask.astype(bool)

            rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
            rc_screen_coords.reshape((-1, 1, 2))

            rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
            rc_mask = rc_mask.astype(bool)

            masked_frame[lc_mask] = 255
            masked_frame[rc_mask] = 255
            return masked_frame
        
        # Left Cheek Only
        case [(1,)]:
            lc_screen_coords = []

            left_cheek_path = create_path(LEFT_CHEEK_IDX)

            # Left cheek screen coordinates
            for cur_source, cur_target in left_cheek_path:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                lc_screen_coords.append((source.get('x'),source.get('y')))
                lc_screen_coords.append((target.get('x'),target.get('y')))
            
            # cv2.fillPoly requires a specific shape and int32 values for the points
            lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
            lc_screen_coords.reshape((-1, 1, 2))

            lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
            lc_mask = lc_mask.astype(bool)

            masked_frame[lc_mask] = 255
            return masked_frame
        
        # Right Cheek Only
        case [(2,)]:
            rc_screen_coords = []
            
            right_cheek_path = create_path(RIGHT_CHEEK_IDX)

            # Right cheek screen coordinates
            for cur_source, cur_target in right_cheek_path:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                rc_screen_coords.append((source.get('x'),source.get('y')))
                rc_screen_coords.append((target.get('x'),target.get('y')))

            # cv2.fillPoly requires a specific shape and int32 values for the points
            rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
            rc_screen_coords.reshape((-1, 1, 2))

            rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
            rc_mask = rc_mask.astype(bool)

            masked_frame[rc_mask] = 255
            return masked_frame

        # Cheeks and Nose
        case [(3,)]:
            lc_screen_coords = []
            rc_screen_coords = []
            nose_screen_coords = []

            left_cheek_path = create_path(LEFT_CHEEK_IDX)
            right_cheek_path = create_path(RIGHT_CHEEK_IDX)

            # Left cheek screen coordinates
            for cur_source, cur_target in left_cheek_path:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                lc_screen_coords.append((source.get('x'),source.get('y')))
                lc_screen_coords.append((target.get('x'),target.get('y')))
            
            # Right cheek screen coordinates
            for cur_source, cur_target in right_cheek_path:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                rc_screen_coords.append((source.get('x'),source.get('y')))
                rc_screen_coords.append((target.get('x'),target.get('y')))
            
            # Nose screen coordinates
            for cur_source, cur_target in NOSE_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                nose_screen_coords.append((source.get('x'),source.get('y')))
                nose_screen_coords.append((target.get('x'),target.get('y')))
            
            lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
            lc_screen_coords.reshape((-1, 1, 2))

            lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
            lc_mask = lc_mask.astype(bool)

            rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
            rc_screen_coords.reshape((-1, 1, 2))

            rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
            rc_mask = rc_mask.astype(bool)

            nose_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            nose_mask = cv.fillConvexPoly(nose_mask, np.array(nose_screen_coords), 1)
            nose_mask = nose_mask.astype(bool)

            masked_frame[lc_mask] = 255
            masked_frame[rc_mask] = 255
            masked_frame[nose_mask] = 255
            return masked_frame
        
        # Both eyes
        case [(4,)]:
            le_screen_coords = []
            re_screen_coords = []

            for cur_source, cur_target in LEFT_EYE_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                le_screen_coords.append((source.get('x'),source.get('y')))
                le_screen_coords.append((target.get('x'),target.get('y')))

            for cur_source, cur_target in RIGHT_EYE_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                re_screen_coords.append((source.get('x'),source.get('y')))
                re_screen_coords.append((target.get('x'),target.get('y')))

            # Creating boolean masks for the facial landmarks 
            le_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
            le_mask = le_mask.astype(bool)

            re_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
            re_mask = re_mask.astype(bool)

            masked_frame[le_mask] = 255
            masked_frame[re_mask] = 255
            return masked_frame

        # Face Skin
        case [(5,)]:
            # Getting screen coordinates of facial landmarks
            le_screen_coords = []
            re_screen_coords = []
            lips_screen_coords = []
            face_outline_coords = []

            # Left eye screen coordinates
            for cur_source, cur_target in LEFT_IRIS_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                le_screen_coords.append((source.get('x'),source.get('y')))
                le_screen_coords.append((target.get('x'),target.get('y')))
            
            # Right eye screen coordinates
            for cur_source, cur_target in RIGHT_IRIS_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                re_screen_coords.append((source.get('x'),source.get('y')))
                re_screen_coords.append((target.get('x'),target.get('y')))

            # Lips screen coordinates
            for cur_source, cur_target in LIPS_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                lips_screen_coords.append((source.get('x'),source.get('y')))
                lips_screen_coords.append((target.get('x'),target.get('y')))
            
            # Face oval screen coordinates
            for cur_source, cur_target in FACE_OVAL_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                face_outline_coords.append((source.get('x'),source.get('y')))
                face_outline_coords.append((target.get('x'),target.get('y')))

            # Creating boolean masks for the facial landmarks 
            le_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
            le_mask = le_mask.astype(bool)

            re_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
            re_mask = re_mask.astype(bool)

            lip_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            lip_mask = cv.fillConvexPoly(lip_mask, np.array(lips_screen_coords), 1)
            lip_mask = lip_mask.astype(bool)

            oval_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            oval_mask = cv.fillConvexPoly(oval_mask, np.array(face_outline_coords), 1)
            oval_mask = oval_mask.astype(bool)

            # Masking the face oval
            masked_frame[oval_mask] = 255
            masked_frame[le_mask] = 0
            masked_frame[re_mask] = 0
            masked_frame[lip_mask] = 0
            return masked_frame
        
        # Chin
        case [(6,)]:
            chin_screen_coords = []
            chin_path = create_path(CHIN_IDX)

            for cur_source, cur_target in chin_path:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                chin_screen_coords.append((source.get('x'), source.get('y')))
                chin_screen_coords.append((target.get('x'), target.get('y')))
            
            chin_screen_coords = np.array(chin_screen_coords, dtype=np.int32)
            chin_screen_coords.reshape((-1, 1, 2))
            
            chin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            chin_mask = cv.fillPoly(img=chin_mask, pts=[chin_screen_coords], color=(255,255,255))
            chin_mask = chin_mask.astype(bool)

            masked_frame[chin_mask] = 255
            return masked_frame

        case _:
            cur_landmark_coords = []
            # Converting landmark coords to screen coords
            for cur_source, cur_target in roi_path:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                cur_landmark_coords.append((source.get('x'),source.get('y')))
                cur_landmark_coords.append((target.get('x'),target.get('y')))

            # Creating boolean masks for the facial landmarks 
            bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
            bool_mask = bool_mask.astype(bool)

            masked_frame[bool_mask] = 255
            return masked_frame

def mask_frame(frame: cv.typing.MatLike, face_mesh:mp.solutions.face_mesh.FaceMesh, mask_type:int,
               background_color:tuple[int,int,int] = (0,0,0), return_mask:bool = False) -> cv.typing.MatLike:

        face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        landmark_screen_coords = []
        background_color = np.asarray(background_color, dtype=np.uint8)

        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:

                # Convert normalised landmark coordinates to x-y pixel coordinates
                for id,lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = frame.shape
                    x,y = int(lm.x * iw), int(lm.y * ih)
                    landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
        else:
            debug_logger.error("Function encountered a FaceNotFoundError attempting to call FaceMesh.process().")
            logger.error("Function encountered a FaceNotFoundError attempting to initialize the mediapipe face landmarker.")
            raise FaceNotFoundError()

        match mask_type:

            case 1: # Face oval mask
                face_outline_coords = []
                
                # face oval screen coordinates
                for cur_source, cur_target in FACE_OVAL_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    face_outline_coords.append((source.get('x'),source.get('y')))
                    face_outline_coords.append((target.get('x'),target.get('y')))

                oval_mask = np.zeros((frame.shape[0],frame.shape[1]))
                oval_mask = cv.fillConvexPoly(oval_mask, np.array(face_outline_coords), 1)
                oval_mask = oval_mask.astype(bool)

                # Otsu thresholding to seperate foreground and background
                grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
                thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

                # Adding a temporary image border to allow for correct floodfill behaviour
                bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
                floodfilled = bordered_thresholded.copy()
                cv.floodFill(floodfilled, None, (0,0), 255)

                # Removing temporary border and creating foreground mask
                floodfilled = floodfilled[10:-10, 10:-10]
                floodfilled = cv.bitwise_not(floodfilled)
                foreground = cv.bitwise_or(thresholded, floodfilled)

                # Masking the face oval
                masked_frame = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                masked_frame[oval_mask] = 255
                if return_mask:
                    return masked_frame
                else:
                    # Remove unwanted background inclusions in the masked area
                    masked_frame = cv.bitwise_and(masked_frame, foreground)
                    masked_frame = np.reshape(masked_frame, (masked_frame.shape[0], masked_frame.shape[1], 1))
                    masked_frame = np.where(masked_frame == 255, frame, background_color)
                    return masked_frame

            case 2: # Face skin isolation
                le_screen_coords = []
                re_screen_coords = []
                lips_screen_coords = []
                face_outline_coords = []

                # left eye screen coordinates
                for cur_source, cur_target in LEFT_EYE_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    le_screen_coords.append((source.get('x'),source.get('y')))
                    le_screen_coords.append((target.get('x'),target.get('y')))
                
                # right eye screen coordinates
                for cur_source, cur_target in RIGHT_EYE_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    re_screen_coords.append((source.get('x'),source.get('y')))
                    re_screen_coords.append((target.get('x'),target.get('y')))

                # lips screen coordinates
                for cur_source, cur_target in MOUTH_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    lips_screen_coords.append((source.get('x'),source.get('y')))
                    lips_screen_coords.append((target.get('x'),target.get('y')))
                
                # face oval screen coordinates
                for cur_source, cur_target in FACE_OVAL_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    face_outline_coords.append((source.get('x'),source.get('y')))
                    face_outline_coords.append((target.get('x'),target.get('y')))

                # Creating boolean masks for the facial regions
                le_mask = np.zeros((frame.shape[0],frame.shape[1]))
                le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                le_mask = le_mask.astype(bool)

                re_mask = np.zeros((frame.shape[0],frame.shape[1]))
                re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                re_mask = re_mask.astype(bool)

                lip_mask = np.zeros((frame.shape[0],frame.shape[1]))
                lip_mask = cv.fillConvexPoly(lip_mask, np.array(lips_screen_coords), 1)
                lip_mask = lip_mask.astype(bool)

                oval_mask = np.zeros((frame.shape[0],frame.shape[1]))
                oval_mask = cv.fillConvexPoly(oval_mask, np.array(face_outline_coords), 1)
                oval_mask = oval_mask.astype(bool)

                # Otsu thresholding to seperate foreground and background
                grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
                thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

                # Adding a temporary image border to allow for correct floodfill behaviour
                bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
                floodfilled = bordered_thresholded.copy()
                cv.floodFill(floodfilled, None, (0,0), 255)

                # Removing temporary border and creating foreground mask
                floodfilled = floodfilled[10:-10, 10:-10]
                floodfilled = cv.bitwise_not(floodfilled)
                foreground = cv.bitwise_or(thresholded, floodfilled)

                # Masking the face oval
                masked_frame = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                masked_frame[oval_mask] = 255

                # Masking out eyes and lips
                masked_frame[le_mask] = 0
                masked_frame[re_mask] = 0
                masked_frame[lip_mask] = 0

                if return_mask:
                    return masked_frame
                else:
                    # Remove unwanted background inclusions in the masked area
                    masked_frame = cv.bitwise_and(masked_frame, foreground)
                    masked_frame = np.reshape(masked_frame, (masked_frame.shape[0], masked_frame.shape[1], 1))
                    masked_frame = np.where(masked_frame == 255, frame, background_color)
                    return masked_frame
            
            case 3: # Both eyes mask
                le_screen_coords = []
                re_screen_coords = []

                # left eye screen coordinates
                for cur_source, cur_target in LEFT_EYE_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    le_screen_coords.append((source.get('x'),source.get('y')))
                    le_screen_coords.append((target.get('x'),target.get('y')))
                
                # right eye screen coordinates
                for cur_source, cur_target in RIGHT_EYE_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    re_screen_coords.append((source.get('x'),source.get('y')))
                    re_screen_coords.append((target.get('x'),target.get('y')))
                
                # Creating boolean masks for the facial regions
                le_mask = np.zeros((frame.shape[0],frame.shape[1]))
                le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                le_mask = le_mask.astype(bool)

                re_mask = np.zeros((frame.shape[0],frame.shape[1]))
                re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                re_mask = re_mask.astype(bool)

                masked_frame = np.zeros((frame.shape[0],frame.shape[1],1), dtype=np.uint8)
                masked_frame[le_mask] = 255
                masked_frame[re_mask] = 255
                
                if return_mask:
                    return masked_frame
                else:
                    masked_frame = np.where(masked_frame == 255, frame, background_color)
                    return masked_frame
            
            case 21: # Both irises mask
                li_screen_coords = []
                ri_screen_coords = []

                # Left iris screen coordinates
                for cur_source, cur_target in LEFT_IRIS_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    li_screen_coords.append((source.get('x'),source.get('y')))
                    li_screen_coords.append((target.get('x'), target.get('y')))
                
                # Right iris screen coordinates
                for cur_source, cur_target in RIGHT_IRIS_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    ri_screen_coords.append((source.get('x'),source.get('y')))
                    ri_screen_coords.append((target.get('x'), target.get('y')))
                
                # Creating boolean masks for the facial regions
                li_mask = np.zeros((frame.shape[0],frame.shape[1]))
                li_mask = cv.fillConvexPoly(li_mask, np.array(li_screen_coords), 1)
                li_mask = li_mask.astype(bool)

                ri_mask = np.zeros((frame.shape[0],frame.shape[1]))
                ri_mask = cv.fillConvexPoly(ri_mask, np.array(ri_screen_coords), 1)
                ri_mask = ri_mask.astype(bool)

                masked_frame = np.zeros((frame.shape[0],frame.shape[1],1), dtype=np.uint8)
                masked_frame[li_mask] = 255
                masked_frame[ri_mask] = 255

                if return_mask:
                    return masked_frame
                else:
                    masked_frame = np.where(masked_frame == 255, frame, background_color)
                    return masked_frame
            
            case 22: # Lips mask
                lips_screen_coords = []

                # Lips screen coordinates
                for cur_source, cur_target in MOUTH_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    lips_screen_coords.append((source.get('x'),source.get('y')))
                    lips_screen_coords.append((target.get('x'),target.get('y')))
                
                # Creating boolean masks for the facial regions
                lip_mask = np.zeros((frame.shape[0],frame.shape[1]))
                lip_mask = cv.fillConvexPoly(lip_mask, np.array(lips_screen_coords), 1)
                lip_mask = lip_mask.astype(bool)

                masked_frame = np.zeros((frame.shape[0],frame.shape[1],1), dtype=np.uint8)
                masked_frame[lip_mask] = 255

                if return_mask:
                    return masked_frame
                else:
                    masked_frame = np.where(masked_frame == 255, frame, background_color)
                    return masked_frame

            case 23: # Hemi-face left mask
                hfl_screen_coords = []

                # Left Hemi-face screen coordinates
                for cur_source, cur_target in HEMI_FACE_LEFT_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    hfl_screen_coords.append((source.get('x'),source.get('y')))
                    hfl_screen_coords.append((target.get('x'),target.get('y')))
                
                # Creating boolean masks for the facial regions
                hfl_mask = np.zeros((frame.shape[0], frame.shape[1]))
                hfl_mask = cv.fillConvexPoly(hfl_mask, np.array(hfl_screen_coords), 1)
                hfl_mask = hfl_mask.astype(bool)

                # Otsu thresholding to seperate foreground and background
                grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
                thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

                # Adding a temporary image border to allow for correct floodfill behaviour
                bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
                floodfilled = bordered_thresholded.copy()
                cv.floodFill(floodfilled, None, (0,0), 255)

                # Removing temporary border and creating foreground mask
                floodfilled = floodfilled[10:-10, 10:-10]
                floodfilled = cv.bitwise_not(floodfilled)
                foreground = cv.bitwise_or(thresholded, floodfilled)

                masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                masked_frame[hfl_mask] = 255

                if return_mask:
                    return masked_frame
                else:
                    # Remove unwanted background inclusions in the masked area
                    masked_frame = cv.bitwise_and(masked_frame, foreground)
                    masked_frame = masked_frame.reshape(masked_frame.shape[0], masked_frame.shape[1], 1)
                    masked_frame = np.where(masked_frame == 255, frame, background_color)
                    return masked_frame

            case 24: # Hemi-face right mask
                hfr_screen_coords = []

                # Right hemi-face screen coordinates
                for cur_source, cur_target in HEMI_FACE_RIGHT_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    hfr_screen_coords.append((source.get('x'),source.get('y')))
                    hfr_screen_coords.append((target.get('x'),target.get('y')))
                
                # Creating boolean masks for the facial regions
                hfr_mask = np.zeros((frame.shape[0], frame.shape[1]))
                hfr_mask = cv.fillConvexPoly(hfr_mask, np.array(hfr_screen_coords), 1)
                hfr_mask = hfr_mask.astype(bool)

                # Otsu thresholding to seperate foreground and background
                grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
                thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

                # Adding a temporary image border to allow for correct floodfill behaviour
                bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
                floodfilled = bordered_thresholded.copy()
                cv.floodFill(floodfilled, None, (0,0), 255)

                # Removing temporary border and creating foreground mask
                floodfilled = floodfilled[10:-10, 10:-10]
                floodfilled = cv.bitwise_not(floodfilled)
                foreground = cv.bitwise_or(thresholded, floodfilled)

                masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                masked_frame[hfr_mask] = 255

                if return_mask:
                    return masked_frame
                else:
                    # Remove unwanted background inclusions in the masked area
                    masked_frame = cv.bitwise_and(masked_frame, foreground)
                    masked_frame = masked_frame.reshape(masked_frame.shape[0], masked_frame.shape[1], 1)
                    masked_frame = np.where(masked_frame == 255, frame, background_color)
                    return masked_frame
            
            case 25: # Hemi-face bottom mask
                hfb_screen_coords = []

                # Bottom hemi-face screen coordinates
                for cur_source, cur_target in HEMI_FACE_BOTTOM_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    hfb_screen_coords.append((source.get('x'),source.get('y')))
                    hfb_screen_coords.append((target.get('x'),target.get('y')))
                
                # Creating boolean masks for the facial regions
                hfb_mask = np.zeros((frame.shape[0], frame.shape[1]))
                hfb_mask = cv.fillConvexPoly(hfb_mask, np.array(hfb_screen_coords), 1)
                hfb_mask = hfb_mask.astype(bool)

                # Otsu thresholding to seperate foreground and background
                grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
                thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

                # Adding a temporary image border to allow for correct floodfill behaviour
                bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
                floodfilled = bordered_thresholded.copy()
                cv.floodFill(floodfilled, None, (0,0), 255)

                # Removing temporary border and creating foreground mask
                floodfilled = floodfilled[10:-10, 10:-10]
                floodfilled = cv.bitwise_not(floodfilled)
                foreground = cv.bitwise_or(thresholded, floodfilled)

                masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                masked_frame[hfb_mask] = 255

                if return_mask:
                    return masked_frame
                else:
                    # Remove unwanted background inclusions in the masked area
                    masked_frame = cv.bitwise_and(masked_frame, foreground)
                    masked_frame = masked_frame.reshape(masked_frame.shape[0], masked_frame.shape[1], 1)
                    masked_frame = np.where(masked_frame == 255, frame, background_color)
                    return masked_frame
            
            case 26: # Hemi-face top mask
                hft_screen_coords = []

                # Bottom hemi-face screen coordinates
                for cur_source, cur_target in HEMI_FACE_TOP_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    hft_screen_coords.append((source.get('x'),source.get('y')))
                    hft_screen_coords.append((target.get('x'),target.get('y')))
                
                # Creating boolean masks for the facial regions
                hft_mask = np.zeros((frame.shape[0], frame.shape[1]))
                hft_mask = cv.fillConvexPoly(hft_mask, np.array(hft_screen_coords), 1)
                hft_mask = hft_mask.astype(bool)

                # Otsu thresholding to seperate foreground and background
                grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
                thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

                # Adding a temporary image border to allow for correct floodfill behaviour
                bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
                floodfilled = bordered_thresholded.copy()
                cv.floodFill(floodfilled, None, (0,0), 255)

                # Removing temporary border and creating foreground mask
                floodfilled = floodfilled[10:-10, 10:-10]
                floodfilled = cv.bitwise_not(floodfilled)
                foreground = cv.bitwise_or(thresholded, floodfilled)

                masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                masked_frame[hft_mask] = 255

                if return_mask:
                    return masked_frame
                else:
                    # Remove unwanted background inclusions in the masked area
                    masked_frame = cv.bitwise_and(masked_frame, foreground)
                    masked_frame = masked_frame.reshape(masked_frame.shape[0], masked_frame.shape[1], 1)
                    masked_frame = np.where(masked_frame == 255, frame, background_color)
                    return masked_frame

            case 14: # Eyes nose mouth mask
                le_screen_coords = []
                re_screen_coords = []
                nose_screen_coords = []
                lips_screen_coords = []

                # left eye screen coordinates
                for cur_source, cur_target in LEFT_EYE_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    le_screen_coords.append((source.get('x'),source.get('y')))
                    le_screen_coords.append((target.get('x'),target.get('y')))
                
                # right eye screen coordinates
                for cur_source, cur_target in RIGHT_EYE_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    re_screen_coords.append((source.get('x'),source.get('y')))
                    re_screen_coords.append((target.get('x'),target.get('y')))
                
                # nose screen coordinates
                for cur_source, cur_target in NOSE_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    nose_screen_coords.append((source.get('x'),source.get('y')))
                    nose_screen_coords.append((target.get('x'),target.get('y')))

                # lips screen coordinates
                for cur_source, cur_target in MOUTH_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    lips_screen_coords.append((source.get('x'),source.get('y')))
                    lips_screen_coords.append((target.get('x'),target.get('y')))

                # Creating boolean masks for the facial regions
                le_mask = np.zeros((frame.shape[0],frame.shape[1]))
                le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                le_mask = le_mask.astype(bool)

                re_mask = np.zeros((frame.shape[0],frame.shape[1]))
                re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                re_mask = re_mask.astype(bool)

                nose_mask = np.zeros((frame.shape[0], frame.shape[1]))
                nose_mask = cv.fillConvexPoly(nose_mask, np.array(nose_screen_coords), 1)
                nose_mask = nose_mask.astype(bool)

                lip_mask = np.zeros((frame.shape[0],frame.shape[1]))
                lip_mask = cv.fillConvexPoly(lip_mask, np.array(lips_screen_coords), 1)
                lip_mask = lip_mask.astype(bool)

                masked_frame = np.zeros((frame.shape[0],frame.shape[1],1), dtype=np.uint8)
                masked_frame[le_mask] = 255
                masked_frame[re_mask] = 255
                masked_frame[nose_mask] = 255
                masked_frame[lip_mask] = 255

                if return_mask:
                    return masked_frame
                else:
                    masked_frame = np.where(masked_frame == 255, frame, background_color)
                    return masked_frame
            
            case _:
                logger.warning("Function encountered a ValueError for input parameter mask_type. "
                               "Please see pyfame_utils.display_face_mask_options() for a full list of predefined mask_types.")
                raise ValueError("Unrecognized value for parameter mask_type.")

def mask_face_region(input_dir:str, output_dir:str, mask_type:int = FACE_OVAL_MASK, with_sub_dirs:bool = False, 
                     background_color: tuple[int,int,int] = (255,255,255), min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    """Applies specified mask type to video files located in input_dir, then outputs masked videos to output_dir.

    Parameters
    ----------

    input_dir: str
        A path string of the directory containing videos to process.

    output_dir: str
        A path string of the directory where processed videos will be written to.

    mask_type: int
        An integer indicating the type of mask to apply to the input videos. For a full list of mask options please see 
        pyfame_utils.display_face_mask_options().

    with_sub_dirs: bool
        Indicates if the input directory contains subfolders.
    
    background_color: tuple of int
        A BGR color code specifying the output files background color.

    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
    
    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
    
    Raises
    ------

    ValueError 
        Given an unknown mask type.
    TypeError 
        Given invalid parameter types.
    OSError: 
        Given invalid path strings for in/output directories
    FaceNotFoundError:
        When mediapipe's face_mesh cannot detect a face.
    FileWriteError:
        On error catches thrown by cv2.imwrite or cv2.VideoWriter.
    FileReadError:
        On error catches thrown by cv2.imread or cv2.VideoCapture.
    """
    
    logger.info("Now entering function mask_face_region().")
    static_image_mode = False
            
    # Type and value checks for function parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: parameter input_dir must be of type str.")
        raise TypeError("Mask_face_region: parameter input_dir must be of type str.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path, or the directory does not exist.")
        raise OSError("Mask_face_region: input directory path is not a valid path, or the directory does not exist.")
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: parameter output_dir must be of type str.")
        raise TypeError("Mask_face_region: parameter output_dir must be of type str.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path, or the directory does not exist.")
        raise OSError("Mask_face_region: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered a ValueError for input parameter output_dir. "
                       "Message: output_dir must be a valid path to a directory.")
        raise ValueError("Mask_face_region: output_dir must be a valid path to a directory.")
    
    if not isinstance(mask_type, int):
        logger.warning("Function encountered a TypeError for input parameter mask_type. "
                       "Message: invalid type for parameter mask_type, expected int.")
        raise TypeError("Mask_face_region: parameter mask_type must be an integer.")
    if mask_type not in MASK_OPTIONS:
        logger.warning("Function encountered a ValueError for input parameter mask_type. "
                       "Message: mask_type must be one of the predefined constants values outlined in pyfame_utils.display_face_mask_options().")
        raise ValueError("Mask_face_region: mask_type must be one of the predefined constants defined in pyfame_utils.display_face_mask_options()")
    
    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a ValueError for input parameter with_sub_dirs. "
                       "Message: with_sub_dirs must be of type bool.")
        raise TypeError("Mask_face_region: parameter with_sub_dirs must be of type bool.")
    
    if not isinstance(background_color, tuple):
        logger.warning("Function encountered a TypeError for input parameter background_color. "
                       "Message: background_color must be of type tuple.")
        raise TypeError("Mask_face_region: parameter background_color must be of type tuple.")
    elif len(background_color) < 3:
        logger.warning("Function encountered a ValueError for input parameter background_color. "
                       "Message: background_color must be a tuple of length 3.")
        raise ValueError("Mask_face_region: parameter background_color expects a length 3 tuple of integers.")
    for val in background_color:
            if not isinstance(val, int):
                logger.warning("Function encountered a ValueError for input parameter background_color. "
                               "Message: background_color must be a tuple of integer values.")
                raise ValueError("Mask_face_region: parameter background_color expects a length 3 tuple of integers.")
    
    background_color = np.asarray(background_color, dtype=np.uint8)

    if not isinstance(min_detection_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_detection_confidence. "
                       "Message: min_detection_confidence must be of type float.")
        raise TypeError("Mask_face_region: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_detection_confidence. "
                       "Message: min_detection_confidence must be a float in the range [0,1].")
        raise ValueError("Mask_face_region: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_tracking_confidence. "
                       "Message: min_tracking_confidence must be of type float.")
        raise TypeError("Mask_face_region: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_tracking_confidence. "
                       "Message: min_tracking_confidence must be a float in the range [0,1].")
        raise ValueError("Mask_face_region: parameter min_tracking_confidence must be in the range [0,1].")
    
    if not isinstance(static_image_mode, bool):
        logger.warning("Function encountered a TypeError for input parameter static_image_mode. "
                       "Message: static_image_mode must be of type bool.")
        raise TypeError("Mask_face_region: parameter static_image_mode must be of type bool.")
    
    # Log input parameters
    mask_type_name = get_variable_name(mask_type, globals())
    logger.info(f"Input parameters: mask_type = {mask_type_name}, background_color = {background_color}.")
    logger.info(f"Mediapipe configurations: min detection confidence = {min_detection_confidence}, "
                f"min tracking confidence = {min_tracking_confidence}, static image mode = {static_image_mode}.")
    
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)

    # Creating a list of file names to iterate through when processing
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
        
    logger.info(f"Function has read {len(files_to_process)} file(s) from input directory {input_dir}.")
    
    # Creating named output directories for video output
    output_dir = create_output_dir(output_dir, "Masked")

    for file in files_to_process:

        # Sniffing input filetype to determine running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        dir_file_path = output_dir + "\\" + filename + "_masked" + extension

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "mp4v"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "mp4v"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                static_image_mode = True
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = True, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case _:
                logger.error("Function has encountered an unparseable file type. " 
                             "Function exiting with status 1. Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                    "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                raise UnrecognizedExtensionError()

        capture = None
        result = None
        
        if not static_image_mode:

            capture = get_video_capture(file)
            size = (int(capture.get(3)), int(capture.get(4)))

            result = get_video_writer(dir_file_path, size, codec=codec)
            
            while True:
                success, frame = capture.read()
                if not success:
                    break
                
                masked_frame = mask_frame(frame, face_mesh, mask_type, background_color)
                result.write(masked_frame)
        
            capture.release()
            result.release()
            logger.info(f"Function execution completed successfully. View outputted file(s) at {dir_file_path}.")
        
        else:
            img = cv.imread(file)
            if img is None:
                raise FileReadError()
            
            masked_img = mask_frame(img, face_mesh, mask_type, background_color)
            success = cv.imwrite(dir_file_path, masked_img)

            if not success:
                logger.error("Function encountered an FileWriteError attempting to call cv2.imwrite(). ")
                debug_logger.error("Function encountered an FileWriteError while attempting to call cv2.imwrite(). " 
                                   f"Ensure output_dir path string is valid, and ensure {file} is not corrupt.")
                raise FileWriteError()
            else:
                logger.info(f"Function execution completed successfully. View outputted file(s) at {output_dir}.")

def occlude_frame(frame:cv.typing.MatLike, mask:cv.typing.MatLike, occlusion_fill:int, landmark_pixel_coords:list[dict]) -> cv.typing.MatLike:
        match occlusion_fill:
            case 8 | 10:
                masked_frame = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
                frame = np.where(masked_frame == 255, 0, frame)
                return frame

            case 9:
                cur_landmark_coords = []
                for cur_source, cur_target in FACE_OVAL_TIGHT_PATH:
                    source = landmark_pixel_coords[cur_source]
                    target = landmark_pixel_coords[cur_target]
                    cur_landmark_coords.append((source.get('x'),source.get('y')))
                    cur_landmark_coords.append((target.get('x'),target.get('y')))

                # Creating boolean masks for the facial landmarks 
                bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
                bool_mask = bool_mask.astype(bool)

                # Extracting the mean pixel value of the face
                bin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                bin_mask[bool_mask] = 255
                mean = cv.mean(frame, bin_mask)
                
                # Fill occlusion regions with facial mean
                masked_frame = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
                mean_img = np.zeros_like(frame, dtype=np.uint8)
                mean_img[:] = mean[:3]
                frame = np.where(masked_frame == 255, mean_img, frame)
                return frame

def occlude_face_region(input_dir:str, output_dir:str, landmarks_to_occlude:list[list[tuple]] = [BOTH_EYES_PATH], occlusion_fill:int = OCCLUSION_FILL_BLACK,
                        with_sub_dirs:bool =  False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    ''' For each video or image contained within the input directory, the landmark regions contained within landmarks_to_occlude 
    will be occluded with either black or the facial mean pixel value. Processed files are then output to Occluded_Video_Output 
    within the specified output directory.

    Parameters
    ----------

    input_dir: str
        A path string to the directory containing files to process. 

    output_dir: str
        A path string to the directory where processed videos will be written.

    landmarks_to_occlude: list of list
        A list of facial landmark paths, either created by the user using utils.create_path(), or selected from the 
        predefined set of facial landmark paths. To see the full list of predefined landmark paths, please see 
        pyfame_utils.display_all_landmark_paths().
    
    occlusion_fill: int
        An integer flag indicating the fill method of the occluded landmark regions. One of OCCLUSION_FILL_BLACK or 
        OCCLUSION_FILL_MEAN. For a full list of available options please see pyfame_utils.display_occlusion_fill_options().
    
    with_sub_dirs: bool
        A boolean flag indicating if the input directory contains subfolders.
    
    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.

    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
        
    Raises
    ------

    TypeError 
        Given invalid parameter types.
    ValueError 
        Given invalid landmark sets or unrecognized fill options.
    OSError 
        Given invalid path strings to either input_dir or output_dir.
    FileReadError
        When an error is encountered instantiating VideoCapture or calling imRead.
    FileWriteError
        When an error is encountered instantiating VideoWriter or calling imWrite.
    UnrecognizedExtensionError
        When an unrecognized image or video file extension is provided.
    '''
    
    static_image_mode = False

    # Performing checks on function parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Occlude_face_region: parameter input_dir must be a str.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path, or the specified directory does not exist.")
        raise OSError("Occlude_face_region: input directory path is not a valid path, or the directory does not exist.")
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Occlude_face_region: parameter output_dir must be a str.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output directory path is not a valid path, or the specified directory does not exist.")
        raise OSError("Occlude_face_region: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered a ValueError for input parameter output_dir. "
                       "Message: output_dir must be a valid path to a directory.")
        raise ValueError("Occlude_face_region: output_dir must be a valid path to a directory.")
    
    if not isinstance(landmarks_to_occlude, list):
        logger.warning("Function encountered a TypeError for input parameter landmarks_to_occlude. "
                       "Message: parameter landmarks_to_occlude expects a list.")
        raise TypeError("Occlude_face_region: parameter landmarks_to_occlude expects a list.")
    for val in landmarks_to_occlude:
            if not isinstance(val, list):
                logger.warning("Function encountered a TypeError for input parameter landmarks_to_occlude. "
                               "Message: parameter landmarks_to_occlude expects a list of list.")
                raise TypeError("Occlude_face_region: landmarks_to_occlude must be a list of lists")
            if not isinstance(val[0], tuple):
                logger.warning("Function encountered a ValueError for input parameter landmarks_to_occlude. "
                               "Message: parameter landmarks_to_occlude expects list[list[tuple]].")
                raise ValueError("Occlude_face_region: landmarks_to_occlude must be a list of list of tuples.")
    
    if not isinstance(occlusion_fill, int):
        logger.warning("Function encountered a TypeError for input parameter occlusion_fill. "
                       "Message: invalid type for parameter occlusion_fill, expected int.")
        raise TypeError("Occlude_face_region: parameter occlusion_fill must be of type int.")
    elif occlusion_fill not in [OCCLUSION_FILL_BLACK, OCCLUSION_FILL_MEAN, OCCLUSION_FILL_BAR]:
        logger.warning("Function encountered a ValueError for input parameter occlusion_fill. "
                       f"Message: {occlusion_fill} is not a valid option for parameter occlusion_fill. "
                       "Please see pyfame_utils.display_occlusion_fill_options().")
        raise ValueError("Occlude_face_region: parameter occlusion_fill must be one of OCCLUSION_FILL_BLACK, OCCLUSION_FILL_MEAN or OCCLUSION_FILL_BAR.")
    if occlusion_fill == OCCLUSION_FILL_BAR:
        print("\nWARNING: OCCLUSION_FILL_BAR is only compatible with BOTH_EYES_PATH, MOUTH_PATH and NOSE_PATH. While the function will occlude"
              + " other paths without error, you may get unexpected behaviour or results.\n")
    
    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Occlude_face_region: parameter with_sub_dirs must be of type bool.")
    
    if not isinstance(min_detection_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_detection_confidence. "
                       "Message: invalid type for parameter min_detection_confidence, expected float.")
        raise TypeError("Occlude_face_region: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_detection_confidence. "
                       "Message: min_detection_confidence must be a float value in the range [0,1].")
        raise ValueError("Occlude_face_region: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_tracking_confidence. "
                       "Message: invalid type for parameter min_tracking_confidence, expected float.")
        raise TypeError("Occlude_face_region: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_tracking_confidence. "
                       "Message: min_tracking_confidence must be a float value in the range [0,1].")
        raise ValueError("Occlude_face_region: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Logging input parameters

    logger.info("Now entering function occlude_face_region().")
    landmark_names_str = "landmarks_to_occlude = ["

    for i in range(len(landmarks_to_occlude)):
        landmark_name = get_variable_name(landmarks_to_occlude[i], globals())

        if i != (len(landmarks_to_occlude) - 1):
            landmark_names_str += f"{landmark_name}, "
        else:
            landmark_names_str += f"{landmark_name}]"
    
    occlusion_fill_type = get_variable_name(occlusion_fill, globals())

    logger.info(f"Input parameters: {landmark_names_str}, occlusion_fill = {occlusion_fill_type}.")
    logger.info(f"Mediapipe configurations: min_detection_confidence = {min_detection_confidence}, "
                f"min_tracking_confidence = {min_tracking_confidence}.")
    
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)

    # Creating a list of file names to iterate through when processing
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
    
    logger.info(f"Function has read {len(files_to_process)} file(s) from input directory {input_dir}.")
    
    output_dir = create_output_dir(output_dir, "Occluded")

    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        dir_file_path = output_dir + f"\\{filename}_occluded{extension}"
        codec = None

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "mp4v"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "mp4v"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                static_image_mode = True
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = True, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case _:
                logger.error("Function has encountered an unparseable file type. " 
                             "Function exiting with status 1. Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                    "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                raise UnrecognizedExtensionError()

        capture = None
        result = None
        min_x_lm = -1
        max_x_lm = -1
        prev_slope = -1

        if not static_image_mode:
            capture = get_video_capture(file)
            size = (int(capture.get(3)), int(capture.get(4)))

            result = get_video_writer(dir_file_path, size, codec=codec)

        while True:

            if static_image_mode:
                frame = cv.imread(file)
                if frame is None:
                    raise FileReadError()
            else:
                success, frame = capture.read()
                if not success:
                    break    

            face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            landmark_screen_coords = []

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:

                    # Convert normalised landmark coordinates to x-y pixel coordinates
                    for id,lm in enumerate(face_landmarks.landmark):
                        ih, iw, ic = frame.shape
                        x,y = int(lm.x * iw), int(lm.y * ih)
                        landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
            else:
                debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process().")
                logger.error("Face mesh detection error, function exiting with status 1.")
                raise FaceNotFoundError()

            masked_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            # Iterate over and mask all provided landmark regions
            for landmark_set in landmarks_to_occlude:

                face_oval_coords = []
                # Converting landmark coords to screen coords
                for cur_source, cur_target in FACE_OVAL_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    face_oval_coords.append((source.get('x'),source.get('y')))
                    face_oval_coords.append((target.get('x'),target.get('y')))
                
                # Creating boolean masks for the face oval
                face_oval_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                face_oval_mask = cv.fillConvexPoly(face_oval_mask, np.array(face_oval_coords), 1)
                face_oval_mask = face_oval_mask.astype(bool)

                max_x = max(face_oval_coords, key=itemgetter(0))[0]
                min_x = min(face_oval_coords, key=itemgetter(0))[0]

                max_y = max(face_oval_coords, key=itemgetter(1))[1]
                min_y = min(face_oval_coords, key=itemgetter(1))[1]

                # Handling special cases (concave landmark regions)
                match landmark_set:
                    # Both Cheeks
                    case [(0,)]:
                        lc_screen_coords = []
                        rc_screen_coords = []

                        left_cheek_path = create_path(LEFT_CHEEK_IDX)
                        right_cheek_path = create_path(RIGHT_CHEEK_IDX)

                        # Left cheek screen coordinates
                        for cur_source, cur_target in left_cheek_path:
                            source = landmark_screen_coords[cur_source]
                            target = landmark_screen_coords[cur_target]
                            lc_screen_coords.append((source.get('x'),source.get('y')))
                            lc_screen_coords.append((target.get('x'),target.get('y')))
                        
                        # Right cheek screen coordinates
                        for cur_source, cur_target in right_cheek_path:
                            source = landmark_screen_coords[cur_source]
                            target = landmark_screen_coords[cur_target]
                            rc_screen_coords.append((source.get('x'),source.get('y')))
                            rc_screen_coords.append((target.get('x'),target.get('y')))
                        
                        lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
                        lc_screen_coords.reshape((-1, 1, 2))

                        lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
                        lc_mask = lc_mask.astype(bool)

                        rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
                        rc_screen_coords.reshape((-1, 1, 2))

                        rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
                        rc_mask = rc_mask.astype(bool)

                        masked_frame[lc_mask] = 255
                        masked_frame[rc_mask] = 255
                        continue
                    
                    # Left Cheek Only
                    case [(1,)]:
                        lc_screen_coords = []

                        left_cheek_path = create_path(LEFT_CHEEK_IDX)

                        # Left cheek screen coordinates
                        for cur_source, cur_target in left_cheek_path:
                            source = landmark_screen_coords[cur_source]
                            target = landmark_screen_coords[cur_target]
                            lc_screen_coords.append((source.get('x'),source.get('y')))
                            lc_screen_coords.append((target.get('x'),target.get('y')))
                        
                        # cv2.fillPoly requires a specific shape and int32 values for the points
                        lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
                        lc_screen_coords.reshape((-1, 1, 2))

                        lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
                        lc_mask = lc_mask.astype(bool)

                        masked_frame[lc_mask] = 255
                        continue
                    
                    # Right Cheek Only
                    case [(2,)]:
                        rc_screen_coords = []
                        
                        right_cheek_path = create_path(RIGHT_CHEEK_IDX)

                        # Right cheek screen coordinates
                        for cur_source, cur_target in right_cheek_path:
                            source = landmark_screen_coords[cur_source]
                            target = landmark_screen_coords[cur_target]
                            rc_screen_coords.append((source.get('x'),source.get('y')))
                            rc_screen_coords.append((target.get('x'),target.get('y')))

                        # cv2.fillPoly requires a specific shape and int32 values for the points
                        rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
                        rc_screen_coords.reshape((-1, 1, 2))

                        rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
                        rc_mask = rc_mask.astype(bool)

                        masked_frame[rc_mask] = 255
                        continue

                    # Cheeks and Nose
                    case [(3,)]:
                        lc_screen_coords = []
                        rc_screen_coords = []
                        nose_screen_coords = []

                        left_cheek_path = create_path(LEFT_CHEEK_IDX)
                        right_cheek_path = create_path(RIGHT_CHEEK_IDX)

                        # Left cheek screen coordinates
                        for cur_source, cur_target in left_cheek_path:
                            source = landmark_screen_coords[cur_source]
                            target = landmark_screen_coords[cur_target]
                            lc_screen_coords.append((source.get('x'),source.get('y')))
                            lc_screen_coords.append((target.get('x'),target.get('y')))
                        
                        # Right cheek screen coordinates
                        for cur_source, cur_target in right_cheek_path:
                            source = landmark_screen_coords[cur_source]
                            target = landmark_screen_coords[cur_target]
                            rc_screen_coords.append((source.get('x'),source.get('y')))
                            rc_screen_coords.append((target.get('x'),target.get('y')))
                        
                        # Nose screen coordinates
                        for cur_source, cur_target in NOSE_PATH:
                            source = landmark_screen_coords[cur_source]
                            target = landmark_screen_coords[cur_target]
                            nose_screen_coords.append((source.get('x'),source.get('y')))
                            nose_screen_coords.append((target.get('x'),target.get('y')))
                        
                        lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
                        lc_screen_coords.reshape((-1, 1, 2))

                        lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
                        lc_mask = lc_mask.astype(bool)

                        rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
                        rc_screen_coords.reshape((-1, 1, 2))

                        rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
                        rc_mask = rc_mask.astype(bool)

                        nose_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        nose_mask = cv.fillConvexPoly(nose_mask, np.array(nose_screen_coords), 1)
                        nose_mask = nose_mask.astype(bool)

                        masked_frame[lc_mask] = 255
                        masked_frame[rc_mask] = 255
                        masked_frame[nose_mask] = 255
                        continue
                    
                    # Both eyes
                    case [(4,)]:

                        if occlusion_fill == OCCLUSION_FILL_BAR:
                            both_eyes_idx = LEFT_IRIS_IDX + RIGHT_IRIS_IDX

                            if min_x_lm < 0 or max_x_lm < 0:
                                min_x = 1000
                                max_x = 0

                                # find the two points closest to the beginning and end x-positions of the landmark region
                                for lm_id in both_eyes_idx:
                                    cur_lm = landmark_screen_coords[lm_id]
                                    if cur_lm.get('x') < min_x:
                                        min_x = cur_lm.get('x')
                                        min_x_lm = lm_id
                                    if cur_lm.get('x') > max_x:
                                        max_x = cur_lm.get('x')
                                        max_x_lm = lm_id
                                
                                # Calculate the slope of the connecting line & angle to the horizontal
                                p1 = landmark_screen_coords[min_x_lm]
                                p2 = landmark_screen_coords[max_x_lm]
                                slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))
                                prev_slope = slope

                                # Compute the center bisecting line of the landmark
                                cx = round((p2.get('y') + p1.get('y'))/2)
                                cy = round((p2.get('x') + p1.get('x'))/2)
                                rot_angle = compute_rot_angle(slope1=slope)
                                
                                rectangle = cv.rectangle(masked_frame, (p1.get('x')-50, cx - 50), (p2.get('x') + 50, cx + 50), (255,255,255), -1)
                                masked_frame_t = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

                                rot_mat = cv.getRotationMatrix2D((cy,cx), rot_angle, 1)
                                rot_img = cv.warpAffine(rectangle, rot_mat, (masked_frame_t.shape[1], masked_frame_t.shape[0]))
                                
                                masked_frame = np.where(rot_img == 255, 255, masked_frame_t)
                                continue

                            else:
                                # Calculate the slope of the connecting line & angle to the horizontal
                                p1 = landmark_screen_coords[min_x_lm]
                                p2 = landmark_screen_coords[max_x_lm]
                                slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))
                                rot_angle = compute_rot_angle(slope1=slope, slope2=prev_slope)
                                prev_slope = slope
                                angle_from_x_axis = compute_rot_angle(slope1=slope)

                                # Compute the center bisecting line of the landmark
                                cx = round((p2.get('y') + p1.get('y'))/2)
                                cy = round((p2.get('x') + p1.get('x'))/2)
                                
                                # Generate the rectangle
                                rectangle = cv.rectangle(masked_frame, (p1.get('x')-50, cx - 50), (p2.get('x') + 50, cx + 50), (255,255,255), -1)
                                masked_frame_t = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                                
                                # Generate rotation matrix and rotate the rectangle
                                rot_mat = cv.getRotationMatrix2D((cy,cx), (rot_angle + angle_from_x_axis), 1)
                                rot_img = cv.warpAffine(rectangle, rot_mat, (masked_frame_t.shape[1], masked_frame_t.shape[0]))
                                
                                masked_frame = np.where(rot_img == 255, 255, masked_frame_t)
                                continue
                        
                        else:
                            le_screen_coords = []
                            re_screen_coords = []

                            for cur_source, cur_target in LEFT_EYE_PATH:
                                source = landmark_screen_coords[cur_source]
                                target = landmark_screen_coords[cur_target]
                                le_screen_coords.append((source.get('x'),source.get('y')))
                                le_screen_coords.append((target.get('x'),target.get('y')))

                            for cur_source, cur_target in RIGHT_EYE_PATH:
                                source = landmark_screen_coords[cur_source]
                                target = landmark_screen_coords[cur_target]
                                re_screen_coords.append((source.get('x'),source.get('y')))
                                re_screen_coords.append((target.get('x'),target.get('y')))

                            # Creating boolean masks for the facial landmarks 
                            le_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                            le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                            le_mask = le_mask.astype(bool)

                            re_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                            re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                            re_mask = re_mask.astype(bool)

                            masked_frame[le_mask] = 255
                            masked_frame[re_mask] = 255
                            continue

                    # Face Skin
                    case [(5,)]:
                        # Getting screen coordinates of facial landmarks
                        le_screen_coords = []
                        re_screen_coords = []
                        lips_screen_coords = []
                        face_outline_coords = []

                        # Left eye screen coordinates
                        for cur_source, cur_target in LEFT_IRIS_PATH:
                            source = landmark_screen_coords[cur_source]
                            target = landmark_screen_coords[cur_target]
                            le_screen_coords.append((source.get('x'),source.get('y')))
                            le_screen_coords.append((target.get('x'),target.get('y')))
                        
                        # Right eye screen coordinates
                        for cur_source, cur_target in RIGHT_IRIS_PATH:
                            source = landmark_screen_coords[cur_source]
                            target = landmark_screen_coords[cur_target]
                            re_screen_coords.append((source.get('x'),source.get('y')))
                            re_screen_coords.append((target.get('x'),target.get('y')))

                        # Lips screen coordinates
                        for cur_source, cur_target in LIPS_PATH:
                            source = landmark_screen_coords[cur_source]
                            target = landmark_screen_coords[cur_target]
                            lips_screen_coords.append((source.get('x'),source.get('y')))
                            lips_screen_coords.append((target.get('x'),target.get('y')))
                        
                        # Face oval screen coordinates
                        for cur_source, cur_target in FACE_OVAL_PATH:
                            source = landmark_screen_coords[cur_source]
                            target = landmark_screen_coords[cur_target]
                            face_outline_coords.append((source.get('x'),source.get('y')))
                            face_outline_coords.append((target.get('x'),target.get('y')))

                        # Creating boolean masks for the facial landmarks 
                        le_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                        le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                        le_mask = le_mask.astype(bool)

                        re_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                        re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                        re_mask = re_mask.astype(bool)

                        lip_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                        lip_mask = cv.fillConvexPoly(lip_mask, np.array(lips_screen_coords), 1)
                        lip_mask = lip_mask.astype(bool)

                        oval_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                        oval_mask = cv.fillConvexPoly(oval_mask, np.array(face_outline_coords), 1)
                        oval_mask = oval_mask.astype(bool)

                        # Masking the face oval
                        masked_frame[oval_mask] = 255
                        masked_frame[le_mask] = 0
                        masked_frame[re_mask] = 0
                        masked_frame[lip_mask] = 0
                        continue
                    
                    # Chin
                    case [(6,)]:
                        chin_screen_coords = []
                        chin_path = create_path(CHIN_IDX)

                        for cur_source, cur_target in chin_path:
                            source = landmark_screen_coords[cur_source]
                            target = landmark_screen_coords[cur_target]
                            chin_screen_coords.append((source.get('x'), source.get('y')))
                            chin_screen_coords.append((target.get('x'), target.get('y')))
                        
                        chin_screen_coords = np.array(chin_screen_coords, dtype=np.int32)
                        chin_screen_coords.reshape((-1, 1, 2))
                        
                        chin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        chin_mask = cv.fillPoly(img=chin_mask, pts=[chin_screen_coords], color=(255,255,255))
                        chin_mask = chin_mask.astype(bool)

                        masked_frame[chin_mask] = 255
                        continue

                    case _:
                        cur_landmark_coords = []
                        # Converting landmark coords to screen coords
                        for cur_source, cur_target in landmark_set:
                            source = landmark_screen_coords[cur_source]
                            target = landmark_screen_coords[cur_target]
                            cur_landmark_coords.append((source.get('x'),source.get('y')))
                            cur_landmark_coords.append((target.get('x'),target.get('y')))
                        
                        if occlusion_fill == OCCLUSION_FILL_BAR:
                            
                            if min_x_lm < 0 or max_x_lm < 0:
                                min_x = 1000
                                max_x = 0

                                # find the two points closest to the beginning and end x-positions of the landmark region
                                unique_landmarks = np.unique(landmark_set)
                                for lm_id in unique_landmarks:
                                    cur_lm = landmark_screen_coords[lm_id]
                                    if cur_lm.get('x') < min_x:
                                        min_x = cur_lm.get('x')
                                        min_x_lm = lm_id
                                    if cur_lm.get('x') > max_x:
                                        max_x = cur_lm.get('x')
                                        max_x_lm = lm_id
                                
                                # Calculate the slope of the connecting line & angle to the horizontal
                                p1 = landmark_screen_coords[min_x_lm]
                                p2 = landmark_screen_coords[max_x_lm]
                                slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))
                                prev_slope = slope

                                # Compute the center bisecting line of the landmark
                                cx = round((p2.get('y') + p1.get('y'))/2)
                                cy = round((p2.get('x') + p1.get('x'))/2)
                                rot_angle = compute_rot_angle(slope1=slope)
                                
                                rectangle = cv.rectangle(masked_frame, (p1.get('x')-50, cx - 50), (p2.get('x') + 50, cx + 50), (255,255,255), -1)
                                masked_frame_t = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

                                rot_mat = cv.getRotationMatrix2D((cy,cx), rot_angle, 1)
                                rot_img = cv.warpAffine(rectangle, rot_mat, (masked_frame_t.shape[1], masked_frame_t.shape[0]))
                                
                                masked_frame = np.where(rot_img == 255, 255, masked_frame_t)
                                continue

                            else:
                                # Calculate the slope of the connecting line & angle to the horizontal
                                p1 = landmark_screen_coords[min_x_lm]
                                p2 = landmark_screen_coords[max_x_lm]
                                slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))
                                prev_slope = slope
                                rot_angle = compute_rot_angle(slope1=slope, slope2=prev_slope)
                                angle_from_x_axis = compute_rot_angle(slope1=prev_slope)

                                # Compute the center bisecting line of the landmark
                                cx = round((p2.get('y') + p1.get('y'))/2)
                                cy = round((p2.get('x') + p1.get('x'))/2)
                                
                                # Generate the rectangle
                                rectangle = cv.rectangle(masked_frame, (p1.get('x')-50, cx - 50), (p2.get('x') + 50, cx + 50), (255,255,255), -1)
                                masked_frame_t = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                                
                                # Generate rotation matrix and rotate the rectangle
                                rot_mat = cv.getRotationMatrix2D((cy,cx), (rot_angle + angle_from_x_axis), 1)
                                rot_img = cv.warpAffine(rectangle, rot_mat, (masked_frame_t.shape[1], masked_frame_t.shape[0]))
                                
                                masked_frame = np.where(rot_img == 255, 255, masked_frame_t)
                                continue

                        else:
                            # Creating boolean masks for the facial landmarks 
                            bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                            bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
                            bool_mask = bool_mask.astype(bool)

                            masked_frame[bool_mask] = 255
                            continue
            
            frame = occlude_frame(frame, masked_frame, occlusion_fill, landmark_screen_coords)

            if static_image_mode:
                success = cv.imwrite(output_dir + "\\" + filename + "_occluded" + extension, frame)
                if not success:
                    logger.error("Function encountered an FileWriteError attempting to call cv2.imwrite(). ")
                    debug_logger.error("Function encountered an FileWriteError while attempting to call cv2.imwrite(). " 
                                      f"Ensure output_dir path string is valid, and ensure {file} is not corrupt.")
                    raise FileWriteError()
                else:
                    break
            else:
                result.write(frame)

        if not static_image_mode:
            capture.release()
            result.release()
        
        logger.info(f"Function execution completed successfully. View outputted file(s) at {output_dir}.")

def blur_face_region(input_dir:str, output_dir:str, blur_method:str | int = "gaussian", mask_type:int = FACE_OVAL_MASK, 
                     k_size:int = 15, with_sub_dirs:bool = False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    """ For each video or image file within `input_dir`, the specified `blur_method` will be applied. Blurred images and video files
    are then written out to `output_dir`.

    Parameters
    ----------

    input_dir: str
        A path string to a directory containing the image or video files to be processed.

    output_dir: str
        A path string to a directory where processed files will be written.

    blur_method: str, int
        Either a string literal ("average", "gaussian", "median"), or a predefined integer constant 
        (BLUR_METHOD_AVERAGE, BLUR_METHOD_GAUSSIAN, BLUR_METHOD_MEDIAN) specifying the type of blurring operation to be performed.
    
    mask_type: int
        An integer flag specifying the type of mask to be applied.
        
    k_size: int
        Specifies the size of the square kernel used in blurring operations. 
    
    with_sub_dirs: bool
        A boolean flag indicating if the input directory contains nested sub-directories.
    
    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.

    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
    
    Raises
    ------

    TypeError:
        Given invalid or incompatible input parameter types.
    ValueError:
        Given an unrecognized value.
    OSError:
        Given invalid path strings to input or output directory.

    """
    static_image_mode = False

    # Performing checks on function parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Blur_face_region: invalid type for parameter input_dir.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path to a directory, or the directory does not exist.")
        raise OSError("Blur_face_region: input directory path is not a valid path, or the directory does not exist.")
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Blur_face_region: parameter output_dir must be a str.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path to a directory, or the directory does not exist.")
        raise OSError("Blur_face_region: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered a ValueError for input parameter output_dir. "
                       "Message: output_dir must be a valid path to a directory.")
        raise ValueError("Blur_face_region: output_dir must be a valid path to a directory.")
    
    if isinstance(blur_method, str):
        if str.lower(blur_method) not in ["average", "gaussian", "median"]:
            logger.warning("Function encountered a ValueError for input parameter blur_method. "
                           "Message: unrecognized value for parameter blur_method.")
            raise ValueError("Blur_face_region: Unrecognised value for parameter blur_method.")
    elif isinstance(blur_method, int):
        if blur_method not in [BLUR_METHOD_AVERAGE, BLUR_METHOD_GAUSSIAN, BLUR_METHOD_MEDIAN]:
            logger.warning("Function encountered a ValueError for input parameter blur_method. "
                           "Message: unrecognized value for parameter blur_method.")
            raise ValueError("Blur_face_region: Unrecognised value for parameter blur_method.")
    else:
        logger.warning("Function encountered a TypeError for input parameter blur_method. "
                       "Message: Invalid type for parameter blur_method, expected int or str.")
        raise TypeError("Blur_face_region: Incompatable type for parameter blur_method.")
    
    if not isinstance(mask_type, int):
        logger.warning("Function encountered a TypeError for input parameter mask_type. "
                       "Message: invalid type for parameter mask_type, expected int.")
        raise TypeError("Blur_face_region: parameter mask_type must be of type int.")
    elif mask_type not in MASK_OPTIONS:
        logger.warning("Function encountered a ValueError for input parameter mask_type. "
                       "Message: unrecognized value for parameter mask_type.")
        raise ValueError("Blur_face_region: unrecognized value for parameter mask_type.")
    
    if not isinstance(k_size, int):
        logger.warning("Function encountered a TypeError for input parameter k_size. "
                       "Message: invalid type for parameter k_size, expected int.")
        raise TypeError("Blur_face_region: parameter k_size must be of type int.")
    elif k_size < 1:
        logger.warning("Function encountered a ValueError for input parameter k_size. "
                       "Message: parameter k_size must be a positive integer (>0).")
        raise ValueError("Blur_face_region: parameter k_size must be a positive integer.")
    
    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Blur_face_region: parameter with_sub_dirs must be of type bool.")
    
    if not isinstance(min_detection_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_detection_confidence. "
                       "Message: invalid type for parameter min_detection_confidence, expected float.")
        raise TypeError("Blur_face_region: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_detection_confidence. "
                       "Message: min_detection_confidence must be a float value in the range [0,1].")
        raise ValueError("Blur_face_region: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_tracking_confidence. "
                       "Message: invalid type for parameter min_tracking_confidence, expected float.")
        raise TypeError("Blur_face_region: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_tracking_confidence. "
                       "Message: min_tracking_confidence must be a float value in the range [0,1].")
        raise ValueError("Blur_face_region: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Logging input parameters

    logger.info("Now entering function blur_face_region().")

    if isinstance(blur_method, str):
        logger.info(f"Input parameters: blur_method = {blur_method}, k_size = {k_size}.")
    else:
        blur_method_name = get_variable_name(blur_method, globals())
        logger.info(f"Input parameters: blur_method = {blur_method_name}, k_size = {k_size}.")

    logger.info(f"Mediapipe configurations: min_detection_confidence = {min_detection_confidence}, "
                f"min_tracking_confidence = {min_tracking_confidence}.")

    
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)

    # Creating a list of file names to iterate through when processing
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
        
    logger.info(f"Function has read {len(files_to_process)} file(s) from input directory {input_dir}.")
    
    output_dir = create_output_dir(output_dir, "Blurred")

    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        dir_file_path = output_dir + f"\\{filename}_blurred{extension}"

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "mp4v"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "mp4v"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                static_image_mode = True
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = True, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case _:
                logger.error("Function has encountered an unparseable file type. " 
                             "Function exiting with status 1. Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                    "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                raise UnrecognizedExtensionError()

        capture = None
        result = None

        if not static_image_mode:
            capture = get_video_capture(file)
            size = (int(capture.get(3)), int(capture.get(4)))

            result = get_video_writer(dir_file_path, size, codec=codec)

        while True:

            if static_image_mode:
                frame = cv.imread(file)
                if frame is None:
                    raise FileReadError()
            else:
                success, frame = capture.read()
                if not success:
                    break    

            masked_frame = mask_frame(frame, face_mesh, mask_type, (0,0,0))

            frame_blurred = None

            match blur_method:
                case "average" | "Average":
                    frame_blurred = cv.blur(frame, (k_size, k_size))
                    frame = np.where(masked_frame != 255, frame_blurred, frame)
                case 11:
                    frame_blurred = cv.blur(frame, (k_size, k_size))
                    frame = np.where(masked_frame != 255, frame_blurred, frame)
                
                case "gaussian" | "Gaussian":
                    frame_blurred = cv.GaussianBlur(frame, (k_size, k_size), 0)
                    frame = np.where(masked_frame != 255, frame_blurred, frame)
                case 12:
                    frame_blurred = cv.GaussianBlur(frame, (k_size, k_size), 0)
                    frame = np.where(masked_frame != 255, frame_blurred, frame)
                
                case "median" | "Median":
                    frame_blurred = cv.medianBlur(frame, k_size)
                    frame = np.where(masked_frame != 255, frame_blurred, frame)
                case 13:
                    frame_blurred = cv.medianBlur(frame, k_size)
                    frame = np.where(masked_frame != 255, frame_blurred, frame)
                
                case _:
                    debug_logger.error("Function encountered a ValueError after parameter checks. "
                                       "Parameter type and value checks may not be performing as intended.")
                    raise ValueError("Unrecognized value for parameter blur_method.")
                    
            if static_image_mode:
                success = cv.imwrite(output_dir + "\\" + filename + "_blurred" + extension, frame)
                if not success:
                    logger.error("Function encountered an FileWriteError attempting to call cv2.imwrite(). ")
                    debug_logger.error("Function encountered an FileWriteError while attempting to call cv2.imwrite(). " 
                                       f"Ensure output_dir path string is valid, and ensure {file} is not corrupt.")
                    raise FileWriteError()
                else:
                    break
            else:
                result.write(frame)

        if not static_image_mode:
            capture.release()
            result.release()
        
        logger.info(f"Function execution completed successfully. View outputted file(s) at {output_dir}.")

def apply_noise(input_dir:str, output_dir:str, noise_method:str|int = "pixelate", pixel_size:int = 32, noise_prob:float = 0.5,
                rand_seed:int | None = None, mean:float = 0.0, standard_dev:float = 0.5, mask_type:int = FACE_OVAL_MASK, 
                with_sub_dirs:bool = False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    """Takes an input image or video file, and applies the specified noise method to the image or each frame of the video. For
    noise_method `pixelate`, an output image size must be specified in order to resize the image/frame's pixels.

    Parameters
    ----------

    input_dir: str
        A path string to a directory containing the video files to be processed.

    output_dir: str
        A path string to a directory where outputted csv files will be written to.

    noise_method: str or int
        Either an integer flag, or string literal specifying the noise method of choice. For the full list of 
        available options, please see pyfame_utils.display_noise_method_options().

    pixel_size: int
        The pixel scale applied when pixelating the output file.
    
    noise_prob: float
        The probability of noise being applied to a given pixel, default is 0.5.
    
    rand_seed: int or None
        A seed for the random number generator used in gaussian and salt and pepper noise. Allows the user 
        to create reproducable results. 
    
    mean: float
        The mean or center of the gaussian distribution used when sampling for gaussian noise.

    standard_dev: float
        The standard deviation or variance of the gaussian distribution used when sampling for gaussian noise.
    
    mask_type: int or None
        An integer specifying the facial region in which to apply the specified noise operation.

    with_sub_dirs: bool
        Indicates whether the input directory contains subfolders.
    
    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.

    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
    
    Raises
    ------

    TypeError: Given invalid parameter typings.
    OSError: Given invalid paths for parameters input_dir or output_dir.
    ValueError: Given an unrecognized noise_method or mask_type.

    """

    logger.info("Now entering function apply_noise().")
    static_image_mode = False

    # Type and value checking input parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Apply_noise: input_dir must be a path string.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path to a directory, or the directory does not exist.")
        raise OSError("Apply_noise: input_dir is not a valid path.")
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Apply_noise: output_dir must be a path string.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path to a directory, or the directory does not exist.")
        raise OSError("Apply_noise: output_dir is not a valid path.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path to a directory, or the directory does not exist.")
        raise ValueError("Apply_noise: output_dir must be a path string to a directory.")
    
    if not isinstance(noise_method, int):
        if not isinstance(noise_method, str):
            logger.warning("Function has encountered a TypeError for input parameter noise_method. " 
                           "Message: invalid type for parameter noise_method, expected int or str.")
            raise TypeError("Apply_noise: parameter noise_method must be either int or str.")
        elif str.lower(noise_method) not in ["salt and pepper", "pixelate", "gaussian"]:
            logger.warning("Function encountered a ValueError for input parameter noise_method. "
                           "Message: unrecognized value for parameter noise_method.")
            raise ValueError("Apply_noise: unrecognized value, please see utils.display_noise_method_options() for the full list of accepted values.")
    elif noise_method not in [NOISE_METHOD_SALT_AND_PEPPER, NOISE_METHOD_PIXELATE, NOISE_METHOD_GAUSSIAN]:
        logger.warning("Function encountered a ValueError for input parameter noise_method. "
                           "Message: unrecognized value for parameter noise_method.")
        raise ValueError("Apply_noise: unrecognized value, please see utils.display_noise_method_options() for the full list of accepted values.")
    
    if not isinstance(pixel_size, int):
        logger.warning("Function encountered a TypeError for input parameter pixel_size. "
                       "Message: invalid type for parameter pixel_size, expected int.")
        raise TypeError("Apply_noise: parameter pixel_size expects an integer.")
    elif pixel_size < 1:
        logger.warning("Function encountered a ValueError for input parameter pixel_size. "
                       "Message: pixel_size must be a positive (>0) integer.")
        raise ValueError("Apply_noise: parameter pixel_size must be a positive integer.")
    
    if not isinstance(noise_prob, float):
        logger.warning("Function encountered a TypeError for input parameter noise_prob. "
                       "Message: invalid type for parameter noise_prob, expected float.")
        raise TypeError("Apply_noise: parameter noise_prob expects a float.")
    elif noise_prob < 0 or noise_prob > 1:
        logger.warning("Function encountered a ValueError for input parameter noise_prob. "
                       "Message: parameter noise_prob must be a float in the range [0,1].")
        raise ValueError("Apply_noise: parameter noise_prob must lie in the range [0,1].")
    
    if rand_seed is not None:
        if not isinstance(rand_seed, int):
            logger.warning("Function encountered a TypeError for input parameter rand_seed. "
                       "Message: invalid type for parameter rand_seed, expected int.")
            raise TypeError("Apply_noise: parameter rand_seed expects an integer.")
    
    if not isinstance(mean, float):
        logger.warning("Function encountered a TypeError for input parameter mean. "
                       "Message: invalid type for parameter mean, expected float.")
        raise TypeError("Apply_noise: parameter mean expects a float.")
    
    if not isinstance(standard_dev, float):
        logger.warning("Function encountered a TypeError for input parameter standard_dev. "
                       "Message: invalid type for parameter standard_dev, expected float.")
        raise TypeError("Apply_noise: parameter standard_dev expects a float.")

    if not isinstance(mask_type, int):
        logger.warning("Function encountered a TypeError for input parameter mask_type. "
                       "Message: invalid type for parameter mask_type, expected int.")
        raise TypeError("Apply_noise: parameter mask_type expects an integer.")
    elif mask_type not in MASK_OPTIONS:
        logger.warning("Function encountered a ValueError for input parameter mask_type. "
                       "Message: unrecognized mask_type. See utils.display_face_mask_options().")
        raise ValueError("Apply_noise: mask_type must be one of the predefined options specified within utils.display_face_mask_options().")
    
    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Apply_noise: parameter with_sub_dirs expects a boolean.")
    
    if not isinstance(min_detection_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_detection_confidence. "
                       "Message: invalid type for parameter min_detection_confidence, expected float.")
        raise TypeError("Apply_noise: parameter min_detection_confidence expects a float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_detection_confidence. "
                       "Message: parameter min_detection_confidence must be a float in the range [0,1].")
        raise ValueError("Apply_noise: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_tracking_confidence. "
                       "Message: invalid type for parameter min_tracking_confidence, expected float.")
        raise TypeError("Apply_noise: parameter min_tracking_confidence expects a float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_tracking_confidence. "
                       "Message: parameter min_tracking_confidence must be a float in the range [0,1].")
        raise ValueError("Apply_noise: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Logging input parameters
    if isinstance(noise_method, str):
        mask_type_name = get_variable_name(mask_type, globals())
        logger.info(f"Input parameters: noise_method = {noise_method}, pixel_size = {pixel_size}, noise_prob = {noise_prob}, "
                    f"rand_seed = {rand_seed}, mean = {mean}, standard_dev = {standard_dev}, mask_type = {mask_type_name}.")
    else:
        noise_method_name = get_variable_name(noise_method, globals())
        mask_type_name = get_variable_name(mask_type, globals())
        logger.info(f"Input parameters: noise_method = {noise_method_name}, pixel_size = {pixel_size}, noise_prob = {noise_prob}, "
                    f"rand_seed = {rand_seed}, mean = {mean}, standard_dev = {standard_dev}, mask_type = {mask_type_name}.")

    logger.info(f"Mediapipe configurations: min_detection_confidence = {min_detection_confidence}, "
                f"min_tracking_confidence = {min_tracking_confidence}.")
            
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
        
    logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")
    
    # Creating named output directories for video output
    output_dir = create_output_dir(output_dir, "Noise_Added")
    
    for file in files_to_process:
            
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        dir_file_path = output_dir + f"\\{filename}_noise_added{extension}"

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "mp4v"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "mp4v"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                static_image_mode = True
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = True, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case _:
                logger.error("Function has encountered an unparseable file type, Function exiting with status 1. " 
                             "Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                    "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                raise UnrecognizedExtensionError()
        
        if not static_image_mode:
            capture = get_video_capture(file)
            size = (int(capture.get(3)), int(capture.get(4)))

            result = get_video_writer(dir_file_path, size, codec=codec)

        # Main Processing loop for video files (will only iterate once over images)
        while True:
            frame = None
            if static_image_mode:
                frame = cv.imread(file)
                if frame is None:
                    logger.error("Function has encountered an error attempting to read in a file. "
                                 f"Message: failed to read in file {file}.")
                    debug_logger.error("Function has encountered an error attempting to call cv2.imread(file). "
                                       f"Message: failed to read in file {file}. The file may be corrupt or incorrectly encoded.")
                    raise FileReadError()
            else:
                success, frame = capture.read()
                if not success:
                    break    

            face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            landmark_screen_coords = []

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:

                    # Convert normalised landmark coordinates to x-y pixel coordinates
                    for id,lm in enumerate(face_landmarks.landmark):
                        ih, iw, ic = frame.shape
                        x,y = int(lm.x * iw), int(lm.y * ih)
                        landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
            else:
                logger.error("Face mesh detection error, function exiting with status 1.")
                debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process() on the current frame.")
                raise FaceNotFoundError()
            
            output_frame = frame.copy()
            if isinstance(noise_method, str):
                noise_method = str.lower(noise_method)

            match noise_method:
                case 'pixelate' | 18:
                    height, width = output_frame.shape[:2]
                    h = frame.shape[0]//pixel_size
                    w = frame.shape[1]//pixel_size

                    temp = cv.resize(frame, (w, h), None, 0, 0, cv.INTER_LINEAR)
                    output_frame = cv.resize(temp, (width, height), None, 0, 0, cv.INTER_NEAREST)

                    img_mask = mask_frame(frame, face_mesh, mask_type)
                    output_frame = np.where(img_mask != 255, output_frame, frame)

                case 'salt and pepper' | 19:
                    # Divide prob in 2 for "salt" and "pepper"
                    thresh = noise_prob
                    noise_prob = noise_prob/2
                    rng = None

                    if rand_seed is not None:
                        rng = np.random.default_rng(rand_seed)
                    else:
                        rng = np.random.default_rng()
                    
                    # Use numpy's random number generator to generate a random matrix in the shape of the frame
                    rdm = rng.random(output_frame.shape[:2])

                    # Create boolean masks 
                    pepper_mask = rdm < noise_prob
                    salt_mask = (rdm >= noise_prob) & (rdm < thresh)
                    
                    # Apply boolean masks
                    output_frame[pepper_mask] = [0,0,0]
                    output_frame[salt_mask] = [255,255,255]

                    img_mask = mask_frame(frame, face_mesh, mask_type)
                    output_frame = np.where(img_mask != 255, output_frame, frame)
                
                case 'gaussian' | 20:
                    var = standard_dev**2
                    rng = None

                    if rand_seed is not None:
                        rng = np.random.default_rng(rand_seed)
                    else:
                        rng = np.random.default_rng()

                    # scikit-image's random_noise function works with floating point images, need to convert our frame's type
                    output_frame = img_as_float64(output_frame)
                    output_frame = random_noise(image=output_frame, mode='gaussian', rng=rng, mean=mean, var=var)
                    output_frame = img_as_ubyte(output_frame)

                    img_mask = mask_frame(frame, face_mesh, mask_type)
                    output_frame = np.where(img_mask != 255, output_frame, frame)

                case _:
                    logger.warning("Function has encountered an unrecognized value for parameter noise_method during execution, "
                                   "exiting with status 1. Input parameter checks may not be functioning as expected.")
                    raise ValueError("Unrecognized value for parameter noise_method.")

            if not static_image_mode:
                result.write(output_frame)
            else:
                success = cv.imwrite(output_dir + "\\" + filename + "_noise_added" + extension, output_frame)
                if not success:
                    logger.error("Function has encountered an error attempting to call cv2.imwrite(), exiting with status 1.")
                    debug_logger.error("Function has encountered an error attempting to call cv2.imwrite() to directory "
                                       f"{dir_file_path}. Please ensure the output directory path is a valid path in your current working directory tree.")
                    raise FileWriteError()
                break
        
        if not static_image_mode:
            capture.release()
            result.release()
    
        logger.info(f"Function execution completed successfully. View outputted file(s) at {output_dir}.")