from pyfame.utilities.constants import *
from pyfame.landmark.get_landmark_coordinates import get_pixel_coordinates_from_landmark
from pyfame.landmark.facial_landmarks import *
from pyfame.utilities.exceptions import *
import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def mask_from_landmarks(frame:cv.typing.MatLike, landmark_paths:list[list[tuple[int,int]]] | list[tuple[int,int]], landmarker_coordinates) -> cv.typing.MatLike:
    
    masked_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # Extracting all of the relevant landmark pixel coordinates
    lc_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_LEFT_CHEEK)
    rc_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_RIGHT_CHEEK)
    chin_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_CHIN)
    nose_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_NOSE)
    ler_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_LEFT_EYE_REGION)
    rer_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_RIGHT_EYE_REGION)
    le_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_LEFT_EYE)
    re_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_RIGHT_EYE)
    li_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_LEFT_IRIS)
    ri_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_RIGHT_IRIS)
    lips_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_LIPS)
    fo_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_FACE_OVAL)

    if isinstance(landmark_paths[0], list):
        for path in landmark_paths:
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
                
                # Both eye regions
                case [(4,)]:
                    # Creating boolean masks for the facial landmarks 
                    ler_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    ler_mask = cv.fillConvexPoly(ler_mask, np.array(ler_screen_coords), 1)
                    ler_mask = ler_mask.astype(bool)

                    rer_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    rer_mask = cv.fillConvexPoly(rer_mask, np.array(rer_screen_coords), 1)
                    rer_mask = rer_mask.astype(bool)

                    masked_frame[ler_mask] = 255
                    masked_frame[rer_mask] = 255

                # Face Skin
                case [(5,)]:
                    # Creating boolean masks for the facial landmarks 
                    ler_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    ler_mask = cv.fillConvexPoly(ler_mask, np.array(ler_screen_coords), 1)
                    ler_mask = ler_mask.astype(bool)

                    rer_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    rer_mask = cv.fillConvexPoly(rer_mask, np.array(rer_screen_coords), 1)
                    rer_mask = rer_mask.astype(bool)

                    lip_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    lip_mask = cv.fillConvexPoly(lip_mask, np.array(lips_screen_coords), 1)
                    lip_mask = lip_mask.astype(bool)

                    oval_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    oval_mask = cv.fillConvexPoly(oval_mask, np.array(fo_screen_coords), 1)
                    oval_mask = oval_mask.astype(bool)

                    # Masking the face oval
                    masked_frame[oval_mask] = 255
                    masked_frame[ler_mask] = 0
                    masked_frame[rer_mask] = 0
                    masked_frame[lip_mask] = 0
                
                # Chin
                case [(6,)]:    
                    chin_screen_coords = np.array(chin_screen_coords, dtype=np.int32)
                    chin_screen_coords.reshape((-1, 1, 2))
                    
                    chin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    chin_mask = cv.fillPoly(img=chin_mask, pts=[chin_screen_coords], color=(255,255,255))
                    chin_mask = chin_mask.astype(bool)

                    masked_frame[chin_mask] = 255
                
                # Lips
                case [(7,)]:
                    lips_screen_coords = np.array(lips_screen_coords, dtype=np.int32)
                    lips_screen_coords.reshape((-1, 1, 2))

                    lips_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    lips_mask = cv.fillPoly(img=lips_mask, pts=[lips_screen_coords], color=(255,255,255))
                    lips_mask = lips_mask.astype(bool)

                    masked_frame[lips_mask] = 255
                
                # Both eyes (sclera)
                case [(8,)]:
                    le_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                    le_mask = le_mask.astype(bool)

                    re_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                    re_mask = re_mask.astype(bool)

                    masked_frame[le_mask] = 255
                    masked_frame[re_mask] = 255
                
                # Both irises
                case [(9,)]:
                    li_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    li_mask = cv.fillConvexPoly(li_mask, np.array(li_screen_coords), 1)
                    li_mask = li_mask.astype(bool)

                    ri_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    ri_mask = cv.fillConvexPoly(ri_mask, np.array(ri_screen_coords), 1)
                    ri_mask = ri_mask.astype(bool)

                    masked_frame[li_mask] = 255
                    masked_frame[ri_mask] = 255

                case _:
                    cur_landmark_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, path)

                    # Creating boolean masks for the facial landmarks 
                    bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
                    bool_mask = bool_mask.astype(bool)

                    masked_frame[bool_mask] = 255
    else:
        match landmark_paths:
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
                ler_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                ler_mask = cv.fillConvexPoly(ler_mask, np.array(ler_screen_coords), 1)
                ler_mask = ler_mask.astype(bool)

                rer_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                rer_mask = cv.fillConvexPoly(rer_mask, np.array(rer_screen_coords), 1)
                rer_mask = rer_mask.astype(bool)

                masked_frame[ler_mask] = 255
                masked_frame[rer_mask] = 255

            # Face Skin
            case [(5,)]:
                # Creating boolean masks for the facial landmarks 
                ler_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                ler_mask = cv.fillConvexPoly(ler_mask, np.array(ler_screen_coords), 1)
                ler_mask = ler_mask.astype(bool)

                rer_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                rer_mask = cv.fillConvexPoly(rer_mask, np.array(rer_screen_coords), 1)
                rer_mask = rer_mask.astype(bool)

                lip_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                lip_mask = cv.fillConvexPoly(lip_mask, np.array(lips_screen_coords), 1)
                lip_mask = lip_mask.astype(bool)

                oval_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                oval_mask = cv.fillConvexPoly(oval_mask, np.array(fo_screen_coords), 1)
                oval_mask = oval_mask.astype(bool)

                # Masking the face oval
                masked_frame[oval_mask] = 255
                masked_frame[ler_mask] = 0
                masked_frame[rer_mask] = 0
                masked_frame[lip_mask] = 0
            
            # Chin
            case [(6,)]:    
                chin_screen_coords = np.array(chin_screen_coords, dtype=np.int32)
                chin_screen_coords.reshape((-1, 1, 2))
                
                chin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                chin_mask = cv.fillPoly(img=chin_mask, pts=[chin_screen_coords], color=(255,255,255))
                chin_mask = chin_mask.astype(bool)

                masked_frame[chin_mask] = 255
            
            # Lips
            case [(7,)]:
                lips_screen_coords = np.array(lips_screen_coords, dtype=np.int32)
                lips_screen_coords.reshape((-1, 1, 2))

                lips_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                lips_mask = cv.fillPoly(img=lips_mask, pts=[lips_screen_coords], color=(255,255,255))
                lips_mask = lips_mask.astype(bool)

                masked_frame[lips_mask] = 255

            # Both eyes (sclera)
            case [(8,)]:
                le_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                le_mask = le_mask.astype(bool)

                re_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                re_mask = re_mask.astype(bool)

                masked_frame[le_mask] = 255
                masked_frame[re_mask] = 255
            
            # Both irises
            case [(9,)]:
                li_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                li_mask = cv.fillConvexPoly(li_mask, np.array(li_screen_coords), 1)
                li_mask = li_mask.astype(bool)

                ri_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                ri_mask = cv.fillConvexPoly(ri_mask, np.array(ri_screen_coords), 1)
                ri_mask = ri_mask.astype(bool)

                masked_frame[li_mask] = 255
                masked_frame[ri_mask] = 255

            case _:
                cur_landmark_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, landmark_paths)

                # Creating boolean masks for the facial landmarks 
                bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
                bool_mask = bool_mask.astype(bool)

                masked_frame[bool_mask] = 255

    return masked_frame.astype(np.uint8)