from pyfame.utilities.constants import *
from pyfame.mesh import get_mesh_coordinates, get_mesh_coordinates_from_path
from pyfame.mesh.mesh_landmarks import *
from pyfame.utilities.exceptions import *
import cv2 as cv
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def mask_from_path(frame:cv.typing.MatLike, region_of_interest:list[list[tuple[int,int]]] | list[tuple[int,int]], face_mesh:mp.solutions.face_mesh.FaceMesh) -> cv.typing.MatLike:
    
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    landmark_screen_coords = get_mesh_coordinates(frame_rgb, face_mesh)
    masked_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # Pre-declaring screen coords
    left_cheek_path = create_path(LEFT_CHEEK_IDX)
    right_cheek_path = create_path(RIGHT_CHEEK_IDX)
    chin_path = create_path(CHIN_IDX)

    lc_screen_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, left_cheek_path)
    rc_screen_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, right_cheek_path)
    chin_screen_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, chin_path)
    nose_screen_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, NOSE_PATH)
    le_screen_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, LEFT_EYE_PATH)
    re_screen_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, RIGHT_EYE_PATH)
    lips_screen_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, MOUTH_PATH)
    fo_screen_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, FACE_OVAL_PATH)

    if isinstance(region_of_interest[0], list):
        for path in region_of_interest:
            match path:
                # Both Cheeks
                case [(0,)]:
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
                
                # Left Cheek Only
                case [(1,)]:
                    # cv2.fillPoly requires a specific shape and int32 values for the points
                    lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
                    lc_screen_coords.reshape((-1, 1, 2))

                    lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
                    lc_mask = lc_mask.astype(bool)

                    masked_frame[lc_mask] = 255
                
                # Right Cheek Only
                case [(2,)]:
                    # cv2.fillPoly requires a specific shape and int32 values for the points
                    rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
                    rc_screen_coords.reshape((-1, 1, 2))

                    rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
                    rc_mask = rc_mask.astype(bool)

                    masked_frame[rc_mask] = 255

                # Cheeks and Nose
                case [(3,)]: 
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
                
                # Both eyes
                case [(4,)]:
                    # Creating boolean masks for the facial landmarks 
                    le_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                    le_mask = le_mask.astype(bool)

                    re_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                    re_mask = re_mask.astype(bool)

                    masked_frame[le_mask] = 255
                    masked_frame[re_mask] = 255

                # Face Skin
                case [(5,)]:
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
                    oval_mask = cv.fillConvexPoly(oval_mask, np.array(fo_screen_coords), 1)
                    oval_mask = oval_mask.astype(bool)

                    # Masking the face oval
                    masked_frame[oval_mask] = 255
                    masked_frame[le_mask] = 0
                    masked_frame[re_mask] = 0
                    masked_frame[lip_mask] = 0
                
                # Chin
                case [(6,)]:    
                    chin_screen_coords = np.array(chin_screen_coords, dtype=np.int32)
                    chin_screen_coords.reshape((-1, 1, 2))
                    
                    chin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    chin_mask = cv.fillPoly(img=chin_mask, pts=[chin_screen_coords], color=(255,255,255))
                    chin_mask = chin_mask.astype(bool)

                    masked_frame[chin_mask] = 255

                case _:
                    cur_landmark_coords = []
                    # Converting landmark coords to screen coords
                    for cur_source, cur_target in path:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        cur_landmark_coords.append((source.get('x'),source.get('y')))
                        cur_landmark_coords.append((target.get('x'),target.get('y')))

                    # Creating boolean masks for the facial landmarks 
                    bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
                    bool_mask = bool_mask.astype(bool)

                    masked_frame[bool_mask] = 255
    else:
        match region_of_interest:
            # Both Cheeks
            case [(0,)]:
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
            
            # Left Cheek Only
            case [(1,)]:
                # cv2.fillPoly requires a specific shape and int32 values for the points
                lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
                lc_screen_coords.reshape((-1, 1, 2))

                lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
                lc_mask = lc_mask.astype(bool)

                masked_frame[lc_mask] = 255
            
            # Right Cheek Only
            case [(2,)]:
                # cv2.fillPoly requires a specific shape and int32 values for the points
                rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
                rc_screen_coords.reshape((-1, 1, 2))

                rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
                rc_mask = rc_mask.astype(bool)

                masked_frame[rc_mask] = 255

            # Cheeks and Nose
            case [(3,)]: 
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
            
            # Both eyes
            case [(4,)]:
                # Creating boolean masks for the facial landmarks 
                le_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                le_mask = le_mask.astype(bool)

                re_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                re_mask = re_mask.astype(bool)

                masked_frame[le_mask] = 255
                masked_frame[re_mask] = 255

            # Face Skin
            case [(5,)]:
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
                oval_mask = cv.fillConvexPoly(oval_mask, np.array(fo_screen_coords), 1)
                oval_mask = oval_mask.astype(bool)

                # Masking the face oval
                masked_frame[oval_mask] = 255
                masked_frame[le_mask] = 0
                masked_frame[re_mask] = 0
                masked_frame[lip_mask] = 0
            
            # Chin
            case [(6,)]:    
                chin_screen_coords = np.array(chin_screen_coords, dtype=np.int32)
                chin_screen_coords.reshape((-1, 1, 2))
                
                chin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                chin_mask = cv.fillPoly(img=chin_mask, pts=[chin_screen_coords], color=(255,255,255))
                chin_mask = chin_mask.astype(bool)

                masked_frame[chin_mask] = 255

            case _:
                cur_landmark_coords = []
                # Converting landmark coords to screen coords
                for cur_source, cur_target in region_of_interest:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    cur_landmark_coords.append((source.get('x'),source.get('y')))
                    cur_landmark_coords.append((target.get('x'),target.get('y')))

                # Creating boolean masks for the facial landmarks 
                bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
                bool_mask = bool_mask.astype(bool)

                masked_frame[bool_mask] = 255

    return masked_frame.astype(np.uint8)