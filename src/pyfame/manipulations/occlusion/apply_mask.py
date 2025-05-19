from pyfame.utils.predefined_constants import *
from pyfame.mesh.landmarks import *
from pyfame.utils.utils import get_variable_name
from pyfame.utils.exceptions import *
from pyfame.io import *
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
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