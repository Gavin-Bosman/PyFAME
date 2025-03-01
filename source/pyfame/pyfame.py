import os
import cv2 as cv
import cv2.typing
import mediapipe as mp
import numpy as np
from skimage.util import *
import sys
from typing import Callable
from .pyfame_utils import *
from operator import itemgetter
import logging
import logging.config
import yaml
import itertools

# Function to initialize loggers
def setup_logging(default_path:str = "config/log_config.yaml"):
    with open(default_path, "rt") as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

setup_logging()
logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def mask_face_region(input_dir:str, output_dir:str, mask_type:int = FACE_OVAL_MASK, with_sub_dirs:bool = False, background_color: tuple[int] = (255,255,255),
                     min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5, static_image_mode:bool = False) -> None:
    """Applies specified mask type to video files located in input_dir, then outputs masked videos to output_dir.

    Parameters
    ----------

    input_dir: str
        A path string of the directory containing videos to process.

    output_dir: str
        A path string of the directory where processed videos will be written to.

    mask_type: int
        An integer indicating the type of mask to apply to the input videos. For a full list of mask options please see
        pyfameutils.MASK_OPTIONS.

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
    
    static_image_mode: bool
        A boolean flag indicating to the mediapipe FaceMesh that it is working with static images rather than
        video frames.
    
    Raises
    ------

    ValueError 
        Given an unknown mask type.
    TypeError 
        Given invalid parameter types.
    OSError: 
        Given invalid path strings for in/output directories
    """

    singleFile = False
    static_image_mode = False

    def process_frame(frame: cv.typing.MatLike, mask_type: int) -> cv.typing.MatLike:

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
            sys.exit(1)

        logger.info("Creating specified image mask.")
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
                for cur_source, cur_target in LIPS_PATH:
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

                masked_frame = np.where(masked_frame == 255, frame, background_color)
                return masked_frame
            
            case 22: # Lips mask
                lips_screen_coords = []

                # Lips screen coordinates
                for cur_source, cur_target in LIPS_PATH:
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

                masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                masked_frame[hfl_mask] = 255

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

                masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                masked_frame[hfr_mask] = 255

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

                masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                masked_frame[hfb_mask] = 255

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

                masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                masked_frame[hft_mask] = 255

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
                for cur_source, cur_target in LIPS_PATH:
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

                masked_frame = np.where(masked_frame == 255, frame, background_color)
                return masked_frame
            
            case _:
                logger.warning("Undefined mask_type, Function exiting with status 1. "
                            "Please see pyfameutils.py for a full list of predefined mask_types.")
                sys.exit(1)
            
    # Type and value checks for function parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: parameter input_dir must be of type str.")
        raise TypeError("Mask_face_region: parameter input_dir must be of type str.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path, or the directory does not exist.")
        raise OSError("Mask_face_region: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: parameter output_dir must be of type str.")
        raise TypeError("Mask_face_region: parameter output_dir must be of type str.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path, or the directory does not exist.")
        raise OSError("Mask_face_region: output directory path is not a valid path, or the directory does not exist.")
    
    if mask_type not in MASK_OPTIONS:
        logger.warning("Function encountered a ValueError for input parameter mask_type. "
                       "Message: mask_type must be one of the predefined constants defined by pyfameutils.MASK_OPTIONS.")
        raise ValueError("Mask_face_region: mask_type must be one of the predefined constants defined within pyfameutils.MASK_OPTIONS")
    
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

    logger.info("Now entering function mask_face_region().")
    logger.info(f"Input parameters: mask_type = {mask_type}, background_color = {background_color}.")
    logger.info(f"Mediapipe configurations: min detection confidence = {min_detection_confidence}, "
                f"min tracking confidence = {min_tracking_confidence}, static image mode = {static_image_mode}.")
    
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)

    # Creating a list of file names to iterate through when processing
    files_to_process = []
    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
        
    logger.info(f"Function has read {len(files_to_process)} file(s) from input directory {input_dir}.")
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Masked"):
        os.mkdir(output_dir + "\\Masked")
        output_dir = output_dir + "\\Masked"
        logger.info(f"Created output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\Masked"

    for file in files_to_process:

        # Sniffing input filetype to determine running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        dir_file_path = output_dir + "\\" + filename + "_masked" + extension

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
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
                sys.exit(1)

        capture = None
        result = None
        
        if not static_image_mode:

            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                logger.error("Function encountered an error when attempting to instantiate cv2.VideoCapture. "
                             "Function exiting with status 1.")
                debug_logger.error(f"Function encountered an error when instantiating cv2.VideoCapture with file {file}.")
                sys.exit(1)
            
            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(dir_file_path, cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                logger.error("Function encountered an error when attempting to instantiate cv2.VideoWriter. "
                             "Function exiting with status 1.")
                debug_logger.error(f"Function encountered an error when instantiating cv2.VideoWriter at file location {dir_file_path}.")
                sys.exit(1)
            
            while True:
                success, frame = capture.read()
                if not success:
                    break
                
                masked_frame = process_frame(frame, mask_type)
                result.write(masked_frame)
        
            capture.release()
            result.release()
            logger.info(f"Function execution completed successfully. View outputted file(s) at {dir_file_path}.")
        
        else:
            img = cv.imread(file)
            masked_img = process_frame(img, mask_type)
            success = cv.imwrite(dir_file_path, masked_img)
            if not success:
                logger.error("Function encountered an error attempting to call cv2.imwrite(). "
                             "Function exiting with status 1.")
                debug_logger.error("Function encountered an error while attempting to call cv2.imwrite(). " 
                                   f"Ensure output_dir path string is valid, and ensure {file} is not corrupt.")
                sys.exit(1)
            else:
                logger.info(f"Function execution completed successfully. View outputted file(s) at {output_dir}.")

def occlude_face_region(input_dir:str, output_dir:str, landmarks_to_occlude:list[list[tuple]] = [EYES_NOSE_MOUTH_MASK], occlusion_fill:int = OCCLUSION_FILL_BAR,
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
        predefined set of facial landmark paths.
    
    occlusion_fill: int
        An integer flag indicating the fill method of the occluded landmark regions. One of OCCLUSION_FILL_BLACK or 
        OCCLUSION_FILL_MEAN.
    
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
    '''
    
    singleFile = False
    static_image_mode = False

    def occlude_frame(frame:cv.typing.MatLike, mask:cv.typing.MatLike, occlusion_fill:int) -> cv.typing.MatLike:
        match occlusion_fill:
            case 8 | 10:
                masked_frame = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
                frame = np.where(masked_frame == 255, 0, frame)
                return frame

            case 9:
                cur_landmark_coords = []
                for cur_source, cur_target in FACE_OVAL_TIGHT_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
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

    # Performing checks on function parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Occlude_face_region: parameter input_dir must be a str.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path, or the specified directory does not exist.")
        raise OSError("Occlude_face_region: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
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
    
    if not isinstance(occlusion_fill, int):
        logger.warning("Function encountered a TypeError for input parameter occlusion_fill. "
                       "Message: invalid type for parameter occlusion_fill, expected int.")
        raise TypeError("Occlude_face_region: parameter occlusion_fill must be of type int.")
    elif occlusion_fill not in [OCCLUSION_FILL_BLACK, OCCLUSION_FILL_MEAN, OCCLUSION_FILL_BAR]:
        logger.warning("Function encountered a ValueError for input parameter occlusion_fill. "
                       f"Message: {occlusion_fill} is not a valid option for parameter occlusion_fill.")
        raise ValueError("Occlude_face_region: parameter occlusion_fill must be one of OCCLUSION_FILL_BLACK, OCCLUSION_FILL_MEAN or OCCLUSION_FILL_BAR.")
    if occlusion_fill == OCCLUSION_FILL_BAR:
        print("\nWARNING: OCCLUSION_FILL_BAR is only compatible with BOTH_EYES_PATH, LIPS_PATH and NOSE_PATH. While the function will occlude"
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
    files_to_process = []
    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    logger.info(f"Function has read {len(files_to_process)} file(s) from input directory {input_dir}.")
    
    if not os.path.exists(output_dir + "\\Occluded"):
        os.mkdir(output_dir + "\\Occluded")
        output_dir = output_dir + "\\Occluded"
        logger.info(f"Created output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\Occluded"

    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        dir_file_path = output_dir + f"\\{filename}_occluded{extension}"
        codec = None

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
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
                sys.exit(1)

        capture = None
        result = None
        min_x_lm = -1
        max_x_lm = -1
        prev_slope = -1

        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                logger.error("Function encountered an error while attempting to initialize cv2.VideoCapture() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function encountered an error while attempting instantiating cv2.VideoCapture() object. "
                                   f"File {file} may be corrupt or encoded in an invalid format.")
                sys.exit(1)

            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_occluded" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                logger.error("Function encountered an error while attempting to initialize cv2.VideoWriter() object. "
                             "Function exiting with status 1.")
                debug_logger.error(f"Function encountered an error while attempting to initialize cv2.VideoWriter() object. "
                                   f"Check that {dir_file_path} is a valid path to a file in your system.")
                sys.exit(1)

        while True:

            if static_image_mode:
                frame = cv.imread(file)
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
                if static_image_mode:
                    debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process().")
                    logger.error("Face mesh detection error, function exiting with status 1.")
                    sys.exit(1)
                else: 
                    continue

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
                                rot_angle = calculate_rot_angle(slope1=slope)
                                
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
                                rot_angle = calculate_rot_angle(slope1=slope, slope2=prev_slope)
                                prev_slope = slope
                                angle_from_x_axis = calculate_rot_angle(slope1=slope)

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
                        for cur_source, cur_target in LIPS_TIGHT_PATH:
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
                                rot_angle = calculate_rot_angle(slope1=slope)
                                
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
                                rot_angle = calculate_rot_angle(slope1=slope, slope2=prev_slope)
                                angle_from_x_axis = calculate_rot_angle(slope1=prev_slope)

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
            
            frame = occlude_frame(frame, masked_frame, occlusion_fill)

            if static_image_mode:
                cv.imwrite(output_dir + "\\" + filename + "_occluded" + extension, frame)
                break
            else:
                result.write(frame)

        if not static_image_mode:
            logger.info(f"Function execution completed successfully. View outputted file at {output_dir}.")
            capture.release()
            result.release()
        else:
            logger.info(f"Function execution completed successfully. View outputted file(s) at {output_dir}.")

def blur_face_region(input_dir:str, output_dir:str, blur_method:str | int = "gaussian", k_size:int = 15, with_sub_dirs:bool = False, 
                     min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
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

    singleFile = False
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
    elif os.path.isfile(input_dir):
        singleFile = True
    
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
    
    if not isinstance(k_size, int):
        logger.warning("Function encountered a TypeError for input parameter k_size. "
                       "Message: invalid type for parameter k_size, expected int.")
        raise TypeError("Blur_face_region: parameter k_size must be of type int.")
    
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
    files_to_process = []
    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
        
    logger.info(f"Function has read {len(files_to_process)} file(s) from input directory {input_dir}.")
    
    if not os.path.exists(output_dir + "\\Blurred"):
        os.mkdir(output_dir + "\\Blurred")
        output_dir = output_dir + "\\Blurred"
        logger.info(f"Created output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\Blurred"

    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        dir_file_path = output_dir + f"\\{filename}_blurred{extension}"

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
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
                sys.exit(1)

        capture = None
        result = None

        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv.VideoCapture() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encountered an error attempting to initialize cv.VideoCapture() object. "
                                   f"File {file} may be corrupt or encoded in an invalid format.")
                sys.exit(1)

            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_blurred" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                logger.error("Function encountered an error while attempting to initialize cv2.VideoWriter() object. "
                             "Function exiting with status 1.")
                debug_logger.error(f"Function encountered an error while attempting to initialize cv2.VideoWriter() object. "
                                   f"Check that {dir_file_path} is a valid path to a file in your system.")
                sys.exit(1)

        while True:

            if static_image_mode:
                frame = cv.imread(file)
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
            elif static_image_mode:
                logger.error("Face mesh detection error, function exiting with status 1.")
                debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process() on the current frame.")
                sys.exit(1)
            else: 
                continue

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

            masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
            masked_frame[oval_mask] = 255

            frame_blurred = None

            match blur_method:
                case "average" | "Average":
                    frame_blurred = cv.blur(frame, (k_size, k_size))
                    frame = np.where(masked_frame == 255, frame_blurred, frame)
                case 11:
                    frame_blurred = cv.blur(frame, (k_size, k_size))
                    frame = np.where(masked_frame == 255, frame_blurred, frame)
                
                case "gaussian" | "Gaussian":
                    frame_blurred = cv.GaussianBlur(frame, (k_size, k_size), 0)
                    frame = np.where(masked_frame == 255, frame_blurred, frame)
                case 12:
                    frame_blurred = cv.GaussianBlur(frame, (k_size, k_size), 0)
                    frame = np.where(masked_frame == 255, frame_blurred, frame)
                
                case "median" | "Median":
                    frame_blurred = cv.medianBlur(frame, k_size)
                    frame = np.where(masked_frame == 255, frame_blurred, frame)
                case 13:
                    frame_blurred = cv.medianBlur(frame, k_size)
                    frame = np.where(masked_frame == 255, frame_blurred, frame)
                
                case _:
                    debug_logger.error("Function encountered a ValueError after parameter checks. "
                                       "Parameter type and value checks may not be performing as intended.")
                    sys.exit(1)
                    

            if static_image_mode:
                cv.imwrite(output_dir + "\\" + filename + "_blurred" + extension, frame)
                break
            else:
                result.write(frame)

        if not static_image_mode:
            capture.release()
            result.release()
        
        logger.info(f"Function execution completed successfully. View outputted file(s) at {output_dir}.")

def apply_noise(input_dir:str, output_dir:str, noise_method:str|int = "pixelate", pixel_size:int = 32, noise_prob:float = 0.5,
                rand_seed:int | None = None, mean:float = 0.0, standard_dev:float = 0.5, mask_type:int | None = None, 
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
        Either an integer flag, or string literal specifying the noise method of choice. 

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
    
    singleFile = False
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
    elif os.path.isfile(input_dir):
        singleFile = True
    
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
        raise OSError("Apply_noise: output_dir must be a path string to a directory.")
    
    if not isinstance(noise_method, int):
        if not isinstance(noise_method, str):
            logger.warning("Function has encountered a TypeError for input parameter noise_method. " 
                           "Message: invalid type for parameter noise_method, expected int or str.")
            raise TypeError("Apply_noise: parameter noise_method must be either int or str.")
        elif str.lower(noise_method) not in ["salt and pepper", "pixelate", "gaussian"]:
            logger.warning("Function encountered a ValueError for input parameter noise_method. "
                           "Message: unrecognized value for parameter noise_method.")
            raise ValueError("Apply_noise: parameter noise method must be one of 'salt and pepper', 'pixelate' or 'gaussian'.")
    elif noise_method not in [NOISE_METHOD_SALT_AND_PEPPER, NOISE_METHOD_PIXELATE, NOISE_METHOD_GAUSSIAN]:
        logger.warning("Function encountered a ValueError for input parameter noise_method. "
                           "Message: unrecognized value for parameter noise_method.")
        raise ValueError("Apply_noise: parameter noise_method must be one of 'salt and pepper', 'pixelate' or 'gaussian'.")
    
    if not isinstance(pixel_size, int):
        logger.warning("Function encountered a TypeError for input parameter pixel_size. "
                       "Message: invalid type for parameter pixel_size, expected int.")
        raise TypeError("Apply_noise: parameter pixel_size expects an integer.")
    
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
                       "Message: unrecognized mask_type. See pyfame.pyfame_utils.MASK_OPTIONS.")
        raise ValueError("Apply_noise: mask_type must be one of the predefined options specified within pyfame_utils.MASK_OPTIONS.")
    
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

    logger.info("Now entering function apply_noise().")
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
    
    def mask_frame(frame: cv.typing.MatLike, mask_type: int) -> cv.typing.MatLike:

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
            sys.exit(1)

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

                # Remove unwanted background inclusions in the masked area
                masked_frame = cv.bitwise_and(masked_frame, foreground)
                masked_frame = np.reshape(masked_frame, (masked_frame.shape[0], masked_frame.shape[1], 1))

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
                for cur_source, cur_target in LIPS_PATH:
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
                
                # Remove unwanted background inclusions in the masked area
                masked_frame = cv.bitwise_and(masked_frame, foreground)
                masked_frame = np.reshape(masked_frame, (masked_frame.shape[0], masked_frame.shape[1], 1))
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

                return masked_frame
            
            case 22: # Lips mask
                lips_screen_coords = []

                # Lips screen coordinates
                for cur_source, cur_target in LIPS_PATH:
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

                masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                masked_frame[hfl_mask] = 255

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

                # Remove unwanted background inclusions in the masked area
                masked_frame = cv.bitwise_and(masked_frame, foreground)
                masked_frame = masked_frame.reshape(masked_frame.shape[0], masked_frame.shape[1], 1)
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

                masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                masked_frame[hfr_mask] = 255

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

                # Remove unwanted background inclusions in the masked area
                masked_frame = cv.bitwise_and(masked_frame, foreground)
                masked_frame = masked_frame.reshape(masked_frame.shape[0], masked_frame.shape[1], 1)
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

                masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                masked_frame[hfb_mask] = 255

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

                # Remove unwanted background inclusions in the masked area
                masked_frame = cv.bitwise_and(masked_frame, foreground)
                masked_frame = masked_frame.reshape(masked_frame.shape[0], masked_frame.shape[1], 1)
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

                masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                masked_frame[hft_mask] = 255

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
                for cur_source, cur_target in LIPS_PATH:
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

                return masked_frame
            
            case _:
                logger.warning("Function has encountered an undefined mask_type during execution, system exiting with status 1. "
                               "Ensure that input parameter checks are functioning as expected.")
                sys.exit(1)
            
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = []

    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
        
    logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Noise_Added"):
        os.mkdir(output_dir + "\\Noise_Added")
        output_dir = output_dir + "\\Noise_Added"
        logger.info(f"Function created output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\Noise_Added"
    
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
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
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
                sys.exit(1)
        
        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoCapture() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encounterd an error attempting to initialize cv2.VideoCapture() object "
                                   f"with file {file}. The file may be corrupt or encoded into an unparseable file type.")
                sys.exit(1)
            
            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_noise_added" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                                   f"Please ensure that {dir_file_path} is a valid path in your current working directory tree.")
                sys.exit(1)

        # Main Processing loop for video files (will only iterate once over images)
        while True:
            frame = None
            if static_image_mode:
                frame = cv.imread(file)
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
            elif static_image_mode:
                logger.error("Face mesh detection error, function exiting with status 1.")
                debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process() on the current frame.")
                sys.exit(1)
            else:
                continue
            
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

                    if mask_type != None:
                        img_mask = mask_frame(frame, mask_type)
                        output_frame = np.where(img_mask == 255, output_frame, frame)

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

                    if mask_type != None:
                        img_mask = mask_frame(frame, mask_type)
                        output_frame = np.where(img_mask == 255, output_frame, frame)
                
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

                    if mask_type != None:
                        img_mask = mask_frame(frame, mask_type)
                        output_frame = np.where(img_mask == 255, output_frame, frame)

                case _:
                    logger.warning("Function has encountered an unrecognized value for parameter noise_method during execution, "
                                   "exiting with status 1. Input parameter checks may not be functioning as expected.")
                    sys.exit(1)

            if not static_image_mode:
                result.write(output_frame)
            else:
                success = cv.imwrite(output_dir + "\\" + filename + "_noise_added" + extension, output_frame)
                if not success:
                    logger.error("Function has encountered an error attempting to call cv2.imwrite(), exiting with status 1.")
                    debug_logger.error("Function has encountered an error attempting to call cv2.imwrite() to directory "
                                       f"{dir_file_path}. Please ensure the output directory path is a valid path in your current working directory tree.")
                    sys.exit(1)
                break
        
        if not static_image_mode:
            capture.release()
            result.release()
    
        logger.info(f"Function execution completed successfully. View outputted file(s) at {output_dir}.")

def facial_scramble(input_dir:str, output_dir:str, out_grayscale:bool = False, scramble_method:int = HIGH_LEVEL_GRID_SCRAMBLE, rand_seed:int|None = None, grid_scramble_threshold:int = 2,
                    grid_square_size:int = 40, with_sub_dirs:bool = False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    """ For each input photo or video file, randomly shuffles the face/facial-landmarks. Output files can either be in full
    color, or grayscale. Specify a random seed for reproducable results.

    Parameters
    ----------

    input_dir: str
        A path string to a directory containing the video files to be processed.

    output_dir: str
        A path string to a directory where processed image or video files will be written too.

    out_grayscale: bool
        A boolean flag indicating if the output file should, or should not be converted to grayscale.

    scramble_method: int
        One of LOW_LEVEL_GRID_SCRAMBLE, HIGH_LEVEL_GRID_SCRAMBLE or LANDMARK_SCRAMBLE. 
    
    rand_seed: int
        An integer used to seed the random number generator used in scrambling. 
    
    grid_scramble_threshold: int
        The maximum number of grid positions that a particular grid square can move. When the image grid is randomly scrambled, 
        each grid square can move at most *threshold* positions in the x and y axis.
    
    grid_square_size: int
        The square dimensions of each grid square used in RANDOM_GRID_SCRAMBLE. The default value of 40, represents grid
        squares of size 40 pixels by 40 pixels.

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

    TypeError:
        Given invalid input parameter typings.
    ValueError: 
        Given an unspecified shuffle_method.
    OSError:
        Given invalid pathstrings to input files. 
    """

    static_image_mode = False
    single_file = False

    # Performing checks on input parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Facial_scramble: parameter input_dir expects a string.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid system path string.")
        raise OSError("Facial_scramble: input_dir must be a valid pathstring to a file or directory.")
    elif os.path.isfile(input_dir):
        single_file = True

    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Facial_scramble: parameter output_dir expects a string.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid system path string.")
        raise OSError("Facial_scramble: output_dir must be a valid pathstring to a file or directory.")
    
    if not isinstance(out_grayscale, bool):
        logger.warning("Function encountered a TypeError for input parameter out_grayscale. "
                       "Message: invalid type for parameter out_grayscale, expected bool.")
        raise TypeError("Facial_scramble: parameter out_grayscale expects a boolean.")
    
    if not isinstance(scramble_method, int):
        logger.warning("Function encountered a TypeError for input parameter scramble_method. "
                       "Message: invalid type for parameter scramble_method, expected int.")
        raise TypeError("Facial_scramble: parameter shuffle_method expects an integer.")
    elif scramble_method not in [27, 28, 29, 30]:
        logger.warning("Function encountered a ValueError for input parameter scramble_method. "
                       "Message: unrecognized value for parameter scramble_method, please see pyfame_utils for the full list of accepted values.")
        raise ValueError("Facial_scramble: parameter shuffle_method must be one of LOW_LEVEL_GRID_SCRAMBLE, "
                         "HIGH_LEVEL_GRID_SCRAMBLE or LANDMARK_SCRAMBLE.")
    
    if rand_seed != None:
        if not isinstance(rand_seed, int):
            logger.warning("Function encountered a TypeError for input parameter rand_seed. "
                           "Message: invalid type for parameter rand_seed, expected int.")
            raise TypeError("Facial_scramble: parameter rand_seed expects an integer.")
    
    if not isinstance(grid_scramble_threshold, int):
        logger.warning("Function encountered a TypeError for input parameter grid_scramble_threshold. "
                       "Message: invalid type for parameter grid_scramble_threshold, expected int.")
        raise TypeError("Facial_scramble: parameter grid_scramble_threshold expects an integer")
    
    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Facial_scramble: parameter with_sub_dirs expects a boolean.")
    
    if not isinstance(min_detection_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_detection_confidence. "
                       "Message: invalid type for parameter min_detection_confidence, expected float.")
        raise TypeError("Facial_scramble: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_detection_confidence. "
                       "Message: min_detection_confidence must be a float in the range [0,1].")
        raise ValueError("Facial_scramble: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_tracking_confidence. "
                       "Message: invalid type for parameter min_tracking_confidence, expected float.")
        raise TypeError("Facial_scramble: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_tracking_confidence. "
                       "Message: min_tracking_confidence must be a float in the range [0,1].")
        raise ValueError("Facial_scramble: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Logging input parameters
    logger.info("Now entering function facial_scramble().")

    scramble_method_name = get_variable_name(scramble_method, globals())
    logger.info(f"Input parameters: out_grayscale = {out_grayscale}, scramble_method = {scramble_method_name}, rand_seed = {rand_seed}, "
                f"grid_scramble_threshold = {grid_scramble_threshold}, grid_square_size = {grid_square_size}, with_sub_dirs = {with_sub_dirs}.")
    
    logger.info(f"Mediapipe configurations: min_detection_confidence = {min_detection_confidence}, min_tracking_confidence = {min_tracking_confidence}.")
    
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = []

    if single_file:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")

    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Scrambled"):
        os.mkdir(output_dir + "\\Scrambled")
        output_dir = output_dir + "\\Scrambled"
        logger.info(f"Function created new output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\Scrambled"

    for file in files_to_process:
            
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        frame = None
        shuffled_keys = None
        rot_angles = None
        x_displacements = None
        rng = None
        dir_file_path = output_dir + f"\\{filename}_scrambled{extension}"

        if rand_seed != None:
            rng = np.random.default_rng(rand_seed)
        else:
            rng = np.random.default_rng()

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
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
                sys.exit(1)

        # Initialise videoCapture and videoWriter objects
        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoCapture() object. "
                            "Function exiting with status 1.")
                debug_logger.error("Function has encounterd an error attempting to initialize cv2.VideoCapture() object "
                                f"with file {file}. The file may be corrupt or encoded into an unparseable file type.")
                sys.exit(1)
            
            size = (int(capture.get(3)), int(capture.get(4)))

            if out_grayscale == True:
                result = cv.VideoWriter(output_dir + "\\" + filename + "_scrambled" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size, isColor=False)
            else:
                result = cv.VideoWriter(output_dir + "\\" + filename + "_scrambled" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size, isColor=True)
                
            if not result.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                            "Function exiting with status 1.")
                debug_logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                                f"Please ensure that {dir_file_path} is a valid path in your current working directory tree.")
                sys.exit(1)
            
            success, frame = capture.read()
            if not success:
                logger.error(f"Function has encountered an error attempting to read a frame, file may be corrupt. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encountered an error attempting to read in a frame from file "
                                   f"{file}. file may be corrupt or incorrectly encoded.")
                sys.exit(1)
        else:
            frame = cv.imread(file)

        if scramble_method != LANDMARK_SCRAMBLE:
            # Precomputing shuffled grid positions
            face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            landmark_screen_coords = []

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:

                    # Convert normalised landmark coordinates to x-y pixel coordinates
                    for id,lm in enumerate(face_landmarks.landmark):
                        ih, iw, ic = frame.shape
                        x,y = int(lm.x * iw), int(lm.y * ih)
                        landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
            elif static_image_mode:
                logger.error("Face mesh detection error, function exiting with status 1.")
                debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process() on the current frame.")
                sys.exit(1)
            else:
                continue

            fo_screen_coords = []

            # Face oval screen coordinates
            for cur_source, cur_target in FACE_OVAL_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                fo_screen_coords.append((source.get('x'), source.get('y')))
                fo_screen_coords.append((target.get('x'), target.get('y')))
            
            fo_mask = np.zeros((frame.shape[0], frame.shape[1]))
            fo_mask = cv.fillConvexPoly(fo_mask, np.array(fo_screen_coords), 1)
            fo_mask = fo_mask.astype(bool)

            # Get x and y bounds of the face oval
            max_x = max(fo_screen_coords, key=itemgetter(0))[0]
            min_x = min(fo_screen_coords, key=itemgetter(0))[0]

            max_y = max(fo_screen_coords, key=itemgetter(1))[1]
            min_y = min(fo_screen_coords, key=itemgetter(1))[1]

            height = max_y-min_y
            width = max_x-min_x

            # Calculate x and y padding, ensuring integer
            x_pad = grid_square_size - (width % grid_square_size)
            y_pad = grid_square_size - (height % grid_square_size)

            if x_pad % 2 !=0:
                min_x -= np.floor(x_pad/2)
                max_x += np.ceil(x_pad/2)
            else:
                min_x -= x_pad/2
                max_x += x_pad/2
            
            if y_pad % 2 !=0:
                min_y -= np.floor(y_pad/2)
                max_y += np.ceil(y_pad/2)
            else:
                min_y -= y_pad/2
                max_y += y_pad/2
                
            # Ensure integer
            min_x = int(min_x)
            max_x = int(max_x)
            min_y = int(min_y)
            max_y = int(max_y)

            height = max_y-min_y
            width = max_x-min_x
            cols = int(width/grid_square_size)
            rows = int(height/grid_square_size)

            grid_squares = {}

            # Populate the grid_squares dict with segments of the frame
            for i in range(rows):
                for j in range(cols):
                    grid_squares.update({(i,j):frame[min_y:min_y + grid_square_size, min_x:min_x + grid_square_size]})
                    min_x += grid_square_size
                min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
                min_y += grid_square_size
            
            keys = list(grid_squares.keys())

            # Shuffle the keys of the grid_squares dict
            if scramble_method == LOW_LEVEL_GRID_SCRAMBLE:
                shuffled_keys = keys.copy()
                shuffled_keys = np.array(shuffled_keys, dtype="i,i")
                shuffled_keys = shuffled_keys.reshape((rows, cols))

                ref_keys = keys.copy()
                ref_keys = np.array(ref_keys, dtype="i,i")
                ref_keys = ref_keys.reshape((rows, cols))

                visited_keys = set()

                # Localised threshold based shuffling of the grid
                for y in range(rows):
                    for x in range(cols):
                        if (x,y) in visited_keys:
                            continue
                        else:
                            x_min = max(0, x - grid_scramble_threshold)
                            x_max = min(cols-1, x + grid_scramble_threshold)
                            y_min = max(0, y - grid_scramble_threshold)
                            y_max = min(rows - 1, y + grid_scramble_threshold)

                            valid_new_positions = [
                                (new_x, new_y)
                                for new_x in range(x_min, x_max + 1)
                                for new_y in range(y_min, y_max + 1)
                                if (new_x, new_y) not in visited_keys
                            ]

                            if valid_new_positions:
                                new_x, new_y = rng.choice(valid_new_positions)

                                # Perform the positional swap
                                shuffled_keys[new_y,new_x] = ref_keys[y,x]
                                shuffled_keys[y,x] = ref_keys[new_y, new_x]
                                visited_keys.add((new_x, new_y))
                                visited_keys.add((x,y))
                            else:
                                visited_keys.add((x,y))

                shuffled_keys = list(shuffled_keys.reshape((-1,)))
                # Ensure tuple(int) 
                shuffled_keys = [tuple([int(x), int(y)]) for (x,y) in shuffled_keys]
            else:
                # Scramble the keys of the grid_squares dict
                shuffled_keys = keys.copy()
                rng.shuffle(shuffled_keys)
        else:
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

        # Main Processing loop for video files (will only iterate once over images)
        while True:
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
                continue
            
            output_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            
            match scramble_method:
                case 27 | 28: # Grid scrambling
                    fo_screen_coords = []

                    # Face oval screen coordinates
                    for cur_source, cur_target in FACE_OVAL_PATH:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        fo_screen_coords.append((source.get('x'), source.get('y')))
                        fo_screen_coords.append((target.get('x'), target.get('y')))
                    
                    fo_mask = np.zeros((frame.shape[0], frame.shape[1]))
                    fo_mask = cv.fillConvexPoly(fo_mask, np.array(fo_screen_coords), 1)
                    fo_mask = fo_mask.astype(bool)
                    
                    min_x = min(fo_screen_coords, key=itemgetter(0))[0]
                    min_y = min(fo_screen_coords, key=itemgetter(1))[1]

                    grid_squares = {}

                    # Populate the grid_squares dict with segments of the frame
                    for i in range(rows):
                        for j in range(cols):
                            grid_squares.update({(i,j):frame[min_y:min_y + grid_square_size, min_x:min_x + grid_square_size]})
                            min_x += grid_square_size
                        min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
                        min_y += grid_square_size
                    
                    min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
                    min_y = int(min(fo_screen_coords, key=itemgetter(1))[1])
                    
                    keys = list(grid_squares.keys())

                    # Populate the scrambled grid dict
                    shuffled_grid_squares = {}

                    for old_key, new_key in zip(keys, shuffled_keys):
                        square = grid_squares.get(new_key)
                        shuffled_grid_squares.update({old_key:square})

                    # Fill the output frame with scrambled grid segments
                    for i in range(rows):
                        for j in range(cols):
                            cur_square = shuffled_grid_squares.get((i,j))
                            output_frame[min_y:min_y+grid_square_size, min_x:min_x+grid_square_size] = cur_square
                            min_x += grid_square_size
                        min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
                        min_y += grid_square_size
                    
                    min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
                    min_y = int(min(fo_screen_coords, key=itemgetter(1))[1])

                    # Calculate the slope of the connecting line & angle to the horizontal
                    # landmarks 162, 389 form a paralell line to the x-axis when the face is vertical
                    p1 = landmark_screen_coords[162]
                    p2 = landmark_screen_coords[389]
                    slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))

                    if slope > 0:
                        angle_from_x_axis = (-1)*calculate_rot_angle(slope1=slope)
                    else:
                        angle_from_x_axis = calculate_rot_angle(slope1=slope)
                    cx = min_x + (width/2)
                    cy = min_y + (height/2)

                    # Using the rotation angle, generate a rotation matrix and apply affine transform
                    rot_mat = cv.getRotationMatrix2D((cx,cy), (angle_from_x_axis), 1)
                    output_frame = cv.warpAffine(output_frame, rot_mat, (frame.shape[1], frame.shape[0]))

                    # Ensure grid only overlays the face oval
                    masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                    masked_frame[fo_mask] = 255
                    output_frame = np.where(masked_frame == 255, output_frame, frame)
                    output_frame = output_frame.astype(np.uint8)

                case 29: # Landmark scrambling
                    le_screen_coords = []
                    re_screen_coords = []
                    nose_screen_coords = []
                    lips_screen_coords = []
                    fo_screen_coords = []
                    
                    # Left eye screen coordinates
                    for cur_source, cur_target in LEFT_EYE_PATH:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        le_screen_coords.append((source.get('x'), source.get('y')))
                        le_screen_coords.append((target.get('x'), target.get('y')))
                    
                    # Right eye screen coordinates
                    for cur_source, cur_target in RIGHT_EYE_PATH:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        re_screen_coords.append((source.get('x'), source.get('y')))
                        re_screen_coords.append((target.get('x'), target.get('y')))
                    
                    # Nose screen coordinates
                    for cur_source, cur_target in NOSE_PATH:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        nose_screen_coords.append((source.get('x'), source.get('y')))
                        nose_screen_coords.append((target.get('x'), target.get('y')))
                    
                    # Lips screen coordinates
                    for cur_source, cur_target in LIPS_PATH:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        lips_screen_coords.append((source.get('x'), source.get('y')))
                        lips_screen_coords.append((target.get('x'), target.get('y')))
                    
                    # Face oval screen coordinates
                    for cur_source, cur_target in FACE_OVAL_TIGHT_PATH:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        fo_screen_coords.append((source.get('x'), source.get('y')))
                        fo_screen_coords.append((target.get('x'), target.get('y')))

                    # Creating boolean masks for the facial landmark regions
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

                    fo_mask = np.zeros((frame.shape[0], frame.shape[1]))
                    fo_mask = cv.fillConvexPoly(fo_mask, np.array(fo_screen_coords), 1)
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
                case _:
                    logger.error("Function has encountered an unrecognized value for parameter scramble_method during execution. "
                                 "Function exiting with status 1.")
                    debug_logger.error("Function has encountered an unrecognized value for parameter scramble_method during execution. "
                                       "Input parameter checks may not be performing as expected.")
                    sys.exit(1)

            if not static_image_mode:
                if out_grayscale == True:
                    output_frame = cv.cvtColor(output_frame, cv.COLOR_BGR2GRAY)
                
                result.write(output_frame)
                success, frame = capture.read()
                if not success:
                    break 
            else:
                if out_grayscale == True:
                    output_frame = cv.cvtColor(output_frame, cv.COLOR_BGR2GRAY)

                success = cv.imwrite(output_dir + "\\" + filename + "_scrambled" + extension, output_frame)
                if not success:
                    logger.error("Function has encountered an error attempting to call cv2.imwrite(), exiting with status 1.")
                    debug_logger.error("Function has encountered an error attempting to call cv2.imwrite() to directory "
                                       f"{dir_file_path}. Please ensure that this path is a valid path in your current working directory tree.")
                    sys.exit(1)
                break
        
        if not static_image_mode:
            capture.release()
            result.release()
        
        logger.info(f"Function execution completed successfully, view outputted file(s) at {output_dir}.")

def point_light_display(input_dir:str, output_dir:str, landmark_regions:list[list] = [FACE_OVAL_PATH], point_density:float = 0.5, 
                        show_history:bool = False, history_mode:int = SHOW_HISTORY_ORIGIN, history_window_msec:int = 500, history_color:tuple[int] = (0,0,255),
                        point_color:tuple[int] = (255,255,255), with_sub_dirs:bool = False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    '''Generates a point light display of the face region contained in the input image or video file. Points are either predefined mediapipe FaceMesh points, passed 
    within the input parameter landmark_points, or they are randomly generated if no input points are provided. The number of points can span 1-468 (the total number of 
    landmarks available on the mediapipe FaceMesh), depending on how dense the user would like the resulting point light display to be. 

    Parameters
    ----------

    input_dir: str
        A path string to a valid directory containing the image or video files to be processed.
    
    output_dir: str
        A path string to a valid directory where the output image or video files will be written to.
    
    landmark_regions: list of list
        Landmark regions to be highlighted by the point light display. Any predefined landmark path or landmark path created with 
        pyfameutils.create_path() can be passed.
    
    point_density: float
        The proportion of points in a given landmark region to be displayed in the output point light display. Its value must
        lie in the range [0,1].
    
    show_history: bool
        A boolean flag indicating if the point path history should be drawn on the output file.
    
    history_mode: int
        An integer flag specifying the history display method. One of SHOW_HISTORY_ORIGIN or SHOW_HISTORY_RELATIVE, which will display history
        path vectors from the current positions to the original position or previous positions, respectively.

    history_window_msec: int
        The time duration (in milliseconds) for history path vectors to be displayed when using SHOW_HISTORY_RELATIVE.

    history_color: tuple of int
        The BGR color code representing the color of history path vectors.
    
    point_color: tuple of int
        The BGR color code representing the color of the points in the output point-light display.
    
    with_sub_dirs: bool
        Indicates whether the input directory contains subfolders.
    
    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.

    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
    '''

    single_file = False

    # Perform checks on input parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Point_light_display: parameter input_dir expects a string.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path string to a directory or file.")
        raise OSError("Point_light_display: parameter input_dir is required to be a valid pathstring.")
    if os.path.isfile(input_dir):
        single_file = True
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Point_light_display: parameter output_dir expects a string.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path string to a directory.")
        raise OSError("Point_light_display: parameter output_dir is required to be a valid path.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path string to a directory.")
        raise OSError("Point_light_display: parameter output_dir must be a path string to a directory.")
    
    if not isinstance(landmark_regions, list):
        logger.warning("Function encountered a TypeError for input parameter landmark_regions. "
                       "Message: invalid type for parameter landmark_regions, expected list.")
        raise TypeError("Point_light_display: parameter landmark_regions expects a list of list.")
    elif len(landmark_regions) == 0:
        logger.warning("Function encountered a ValueError for input parameter landmark_regions. "
                       "Message: landmark_regions cannot be an empty list.")
        raise ValueError("Point_light_display: parameter landmark_regions cannot be an empty list.")
    for val in landmark_regions:
        if not isinstance(val, list):
            logger.warning("Function encountered a TypeError for input parameter landmark_regions. "
                    "Message: invalid type for parameter landmark_regions, expected list of list.")
            raise TypeError("Point_light_display: parameter landmark_regions expects a list of list.")
    
    if not isinstance(point_density, float):
        logger.warning("Function encountered a TypeError for input parameter point_density. "
                       "Message: invalid type for parameter point_density, expected float.")
        raise TypeError("Point_light_display: parameter point_density expects a float.")
    elif point_density < 0 or point_density > 1:
        logger.warning("Function encountered a ValueError for input parameter point_density. "
                       "Message: point_density must be a float in the range [0,1].")
        raise ValueError("Point_light_display: parameter point_density must be in the range [0,1].")
    
    if not isinstance(show_history, bool):
        logger.warning("Function encountered a TypeError for input parameter show_history. "
                       "Message: invalid type for parameter show_history, expected bool.")
        raise TypeError("Point_light_display: parameter show_history must be a boolean.")

    if not isinstance(history_mode, int):
        logger.warning("Function encountered a TypeError for input parameter history_mode. "
                       "Message: invalid type for parameter history_mode, expected int.")
        raise TypeError("Point_light_display: parameter history_mode must be an integer.")
    elif history_mode not in [SHOW_HISTORY_ORIGIN, SHOW_HISTORY_RELATIVE]:
        logger.warning("Function encountered a ValueError for input parameter history_mode. "
                       "Message: unrecognized value for parameter history_mode, please see pyfame_utils for a full list of accepted values.")
        raise ValueError("Point_light_display: parameter history_mode must be one of SHOW_HISTORY_ORIGIN or SHOW_HISTORY_RELATIVE.")
    
    if not isinstance(history_window_msec, int):
        logger.warning("Function encountered a TypeError for input parameter history_window_msec. "
                       "Message: invalid type for parameter history_window_msec, expected int.")
        raise TypeError("Point_light_display: parameter history_window_msec must be an integer.")
    elif history_window_msec < 0:
        show_history = False
    
    if not isinstance(point_color, tuple):
        logger.warning("Function encountered a TypeError for input parameter point_color. "
                       "Message: invalid type for parameter point_color, expected tuple[int].")
        raise TypeError("Point_light_display: parameter point_color must be of type tuple.")
    elif len(point_color) < 3:
        logger.warning("Function encountered a ValueError for input parameter point_color. "
                       "Message: point_color must be a length 3 tuple of integers.")
        raise ValueError("Point_light_display: parameter point_color expects a length 3 tuple of integers.")
    for val in point_color:
        if not isinstance(val, int):
            logger.warning("Function encountered a ValueError for input parameter point_color. "
                            "Message: point_color must be a length 3 tuple of integers.")
            raise ValueError("Point_light_display: parameter point_color expects a length 3 tuple of integers.")
    
    if not isinstance(history_color, tuple):
        logger.warning("Function encountered a TypeError for input parameter history_color. "
                       "Message: invalid type for parameter history_color, expected tuple[int].")
        raise TypeError("Point_light_display: parameter history_color must be of type tuple.")
    elif len(history_color) < 3:
        logger.warning("Function encountered a ValueError for input parameter history_color. "
                       "Message: history_color must be a length 3 tuple of integers.")
        raise ValueError("Point_light_display: parameter history_color expects a length 3 tuple of integers.")
    for val in history_color:
        if not isinstance(val, int):
            logger.warning("Function encountered a ValueError for input parameter history_color. "
                            "Message: history_color must be a length 3 tuple of integers.")
            raise ValueError("Point_light_display: parameter history_color expects a length 3 tuple of integers.")

    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Point_light_display: parameter with_sub_dirs must be a boolean.")
    
    if not isinstance(min_detection_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_detection_confidence. "
                       "Message: invalid type for parameter min_detection_confidence, expected float.")
        raise TypeError("Point_light_display: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_detection_confidence. "
                       "Message: min_detection_confidence must be a float in the range [0,1].")
        raise ValueError("Point_light_display: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_tracking_confidence. "
                       "Message: invalid type for parameter min_tracking_confidence, expected float.")
        raise TypeError("Point_light_display: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_tracking_confidence. "
                       "Message: min_tracking_confidence must be a float in the range [0,1].")
        raise ValueError("Point_light_display: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Logging input parameters
    logger.info("Now entering function point_light_display().")

    landmark_region_names = "Input parameters: landmark_regions = ["

    for i in range(len(landmark_regions)):
        lm_region = landmark_regions[i]
        region_name = get_variable_name(lm_region, globals())

        if i == len(landmark_regions) - 1:
            landmark_region_names += f"{region_name}]"
        else:
            landmark_region_names += f"{region_name}, "
    
    hist_mode_name = get_variable_name(history_mode, globals())

    logger.info(f"{landmark_region_names}, point_density = {point_density}, show_history = {show_history}, "
                f"history_mode = {hist_mode_name}, history_window_msec = {history_window_msec}, history_color = {history_color}, "
                f"point_color = {point_color}, with_sub_dirs = {with_sub_dirs}.")
    
    logger.info(f"Mediapipe configurations: min_detection_confidence = {min_detection_confidence}, min_tracking_confidence = {min_tracking_confidence}.")
    
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = []

    if single_file:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
        
    logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\PLD"):
        os.mkdir(output_dir + "\\PLD")
        output_dir = output_dir + "\\PLD"
        logger.info(f"Function created new output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\PLD"

    for file in files_to_process:

        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        dir_file_path = output_dir + f"\\{filename}_pld{extension}"

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = True, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case _:
                logger.error("Function has encountered an unparseable file type, Function exiting with status 1. " 
                             "Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                    "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                sys.exit(1)
        
        capture = cv.VideoCapture(file)
        if not capture.isOpened():
            logger.error("Function has encountered an error attempting to initialize cv2.VideoCapture() object. "
                        "Function exiting with status 1.")
            debug_logger.error("Function has encounterd an error attempting to initialize cv2.VideoCapture() object "
                            f"with file {file}. The file may be corrupt or encoded into an unparseable file type.")
            sys.exit(1)
        
        size = (int(capture.get(3)), int(capture.get(4)))

        result = cv.VideoWriter(output_dir + "\\" + filename + "_pld" + extension,
                                cv.VideoWriter.fourcc(*codec), 30, size)
        if not result.isOpened():
            logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                        "Function exiting with status 1.")
            debug_logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                            f"Please ensure that {dir_file_path} is a valid path in your current working directory tree.")
            sys.exit(1)
        
        # Persistent variables for processing loop
        counter = 0
        lm_idx_to_display = np.array([], dtype=np.uint8)
        prev_points = None

        success, frame = capture.read()
        if not success:
            logger.error(f"Function has encountered an error attempting to read a frame, file may be corrupt. "
                         "Function exiting with status 1.")
            debug_logger.error("Function has encountered an error attempting to read in a frame from file "
                               f"{file}. file may be corrupt or incorrectly encoded.")
            sys.exit(1)

        mask = np.zeros_like(frame, dtype=np.uint8)
        fps = capture.get(cv.CAP_PROP_FPS)
        frame_history_count = round(fps * (history_window_msec/1000))
        frame_history = []

        # Main Processing loop for video files
        while True:

            output_img = np.zeros_like(frame, dtype=np.uint8)

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
                continue
            
            if counter == 0:
                for lm_path in landmark_regions:
                    lm_mask = np.zeros((frame.shape[0], frame.shape[1]))

                    match lm_path:
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

                            lm_mask[lc_mask] = 255
                            lm_mask[rc_mask] = 255
                            lm_mask = lm_mask.astype(bool)
                        
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

                            lm_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            lm_mask = cv.fillPoly(img=lm_mask, pts=[lc_screen_coords], color=(255,255,255))
                            lm_mask = lm_mask.astype(bool)
                        
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

                            lm_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            lm_mask = cv.fillPoly(img=lm_mask, pts=[rc_screen_coords], color=(255,255,255))
                            lm_mask = lm_mask.astype(bool)

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

                            lm_mask[lc_mask] = 255
                            lm_mask[rc_mask] = 255
                            lm_mask[nose_mask] = 255
                            lm_mask = lm_mask.astype(bool)
                        
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

                            lm_mask[le_mask] = 255
                            lm_mask[re_mask] = 255
                            lm_mask = lm_mask.astype(bool)

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
                            lm_mask[oval_mask] = 255
                            lm_mask[le_mask] = 0
                            lm_mask[re_mask] = 0
                            lm_mask[lip_mask] = 0
                            lm_mask = lm_mask.astype(bool)
                        
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
                            
                            lm_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            lm_mask = cv.fillPoly(img=lm_mask, pts=[chin_screen_coords], color=(255,255,255))
                            lm_mask = lm_mask.astype(bool)

                        case _:
                            lm_coords = []

                            for cur_source, cur_target in lm_path:
                                source = landmark_screen_coords[cur_source]
                                target = landmark_screen_coords[cur_target]
                                lm_coords.append((source.get('x'), source.get('y')))
                                lm_coords.append((target.get('x'), target.get('y')))
                            
                            lm_mask = np.zeros((frame.shape[0],frame.shape[1]))
                            lm_mask = cv.fillConvexPoly(lm_mask, np.array(lm_coords), 1)
                            lm_mask = lm_mask.astype(bool)

                    # Use the generated bool mask to get valid indicies
                    for lm in landmark_screen_coords:
                        x = lm.get('x')
                        y = lm.get('y')
                        if lm_mask[y,x] == True:
                            lm_idx_to_display = np.append(lm_idx_to_display, lm.get('id'))
            
                if point_density != 1.0:
                        # Pad and reshape idx array to slices of size 10
                        new_lm_idx = lm_idx_to_display.copy()
                        pad_size = len(new_lm_idx)%10
                        append_arr = np.full(10-pad_size, -1)
                        new_lm_idx = np.append(new_lm_idx, append_arr)
                        new_lm_idx = new_lm_idx.reshape((-1, 10))

                        bin_idx_mask = np.zeros((new_lm_idx.shape[0], new_lm_idx.shape[1]))

                        for i,_slice in enumerate(new_lm_idx):
                            num_ones = round(np.floor(10*point_density))

                            # Generate normal distribution around center of slice
                            mean = 4.5
                            std_dev = 1.67
                            normal_idx = np.random.normal(loc=mean, scale=std_dev, size=num_ones)
                            normal_idx = np.clip(normal_idx, 0, 9).astype(int)

                            new_bin_arr = np.zeros(10)
                            for idx in normal_idx:
                                new_bin_arr[idx] = 1
                            
                            # Ensure the correct proportion of ones are present
                            while new_bin_arr.sum() < num_ones:
                                add_idx = np.random.choice(np.where(new_bin_arr == 0)[0])
                                new_bin_arr[add_idx] = 1
                            
                            bin_idx_mask[i] = new_bin_arr
                        
                        bin_idx_mask = bin_idx_mask.reshape((-1,))
                        new_lm_idx = new_lm_idx.reshape((-1,))
                        lm_idx_to_display = np.where(bin_idx_mask == 1, new_lm_idx, -1)
            
            # After landmark idx are computed in first iteration, this conditional block becomes the main loop
            if counter > 0:

                cur_points = []
                history_mask = np.zeros_like(frame, dtype=np.uint8)

                # Get current landmark screen coords
                for id in lm_idx_to_display:
                    if id != -1:
                        point = landmark_screen_coords[id]
                        cur_points.append((point.get('x'), point.get('y')))
                
                if prev_points == None or show_history == False:
                    prev_points = cur_points.copy()

                    for point in cur_points:
                        x1, y1 = point
                        if x1 > 0 and y1 > 0:
                            output_img = cv.circle(output_img, (x1, y1), 3, point_color, -1)

                elif history_mode == SHOW_HISTORY_ORIGIN:
                    # If show_history is true, display vector paths of all points
                    for (old, new) in zip(prev_points, cur_points):
                        x0, y0 = old
                        x1, y1 = new
                        mask = cv.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), history_color, 2)
                        output_img = cv.circle(output_img, (int(x1), int(y1)), 3, point_color, -1)

                    output_img = cv.add(output_img, mask)
                    mask = np.zeros_like(frame, dtype=np.uint8)

                else:
                    # If show_history is true, display vector paths of all points
                    for (old, new) in zip(prev_points, cur_points):
                        x0, y0 = old
                        x1, y1 = new
                        mask = cv.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), history_color, 2)
                        output_img = cv.circle(output_img, (int(x1), int(y1)), 3, point_color, -1)

                    if len(frame_history) < frame_history_count:
                        frame_history.append(mask)
                        for img in frame_history:
                            history_mask = cv.bitwise_or(history_mask, img)
                    else:
                        frame_history.append(mask)
                        frame_history.pop(0)
                        for img in frame_history:
                            history_mask = cv.bitwise_or(history_mask, img)

                    prev_points = cur_points.copy()
                    output_img = cv.add(output_img, history_mask)
                    mask = np.zeros_like(frame, dtype=np.uint8)

                result.write(output_img)

                success, frame = capture.read()
                if not success:
                    break 
            
            counter += 1
        
        logger.info(f"Function completed execution successfully, view outputted file(s) at {output_dir}.")
        capture.release()
        result.release()

def get_optical_flow(input_dir:str, output_dir:str, optical_flow_type: int|str = SPARSE_OPTICAL_FLOW, landmarks_to_track:list[int]|None = None,
                     max_corners:int = 20, corner_quality_lvl:float = 0.3, min_corner_distance:int = 7, block_size:int = 7, win_size:tuple[int] = (15,15), 
                     max_pyr_lvl:int = 2, pyr_scale:float = 0.5, max_lk_iter:int = 10, lk_accuracy_thresh:float = 0.03, poly_sigma:float = 1.5, 
                     point_color:tuple[int] = (255,255,255), point_radius:int = 5, vector_color:tuple[int]|None = None, with_sub_dirs:bool = False, 
                     csv_sample_freq:int = 1000, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    '''Takes an input video file, and computes the sparse or dense optical flow, outputting the visualised optical flow to output_dir.
    Sparse optical flow uses the Lucas-Kanade algorithm to track a set of sparse points found using the Shi-Tomasi corner points algorithm. Alternatively, 
    the user may provide a set of FaceMesh landmark points to track. Dense optical flow uses Farneback's algorithm to track all points in a frame.

    Parameters
    ----------

    input_dir: str
        A path string to a directory containing the video files to be processed.

    output_dir: str
        A path string to a directory where outputted csv files will be written to.

    optical_flow_type: str or int
        Either an integer specifer (one of SPARSE_OPTICAL_FLOW, DENSE_OPTICAL_FLOW) or a string literal specifying 
        "sparse" or "dense".
    
    landmarks_to_track: list or None
        A list of user provided integer landmark id's to be tracked during Lucas-Kanade optical flow. If no landmarks are 
        provided, the Shi-Tomasi corners algorithm will be used to find good tracking points.
    
    max_corners: int
        The maximum number of corners to detect using the Shi-Tomasi corners algorithm.
    
    corner_quality_lvl: float
        A float in the range [0,1] that determines the minimum quality of accepted corners found using the Shi-Tomasi corners algorithm.
    
    min_corner_distance: int
        The minimum Euclidean distance required between two detected corners for both corners to be accepted.
    
    block_size: int
        The size of the search window used in the Shi-Tomasi corners algorithm (for sparse optical flow), or the size of the pixel neighborhood
        used in Farneback's dense optical flow algorithm.
    
    win_size: tuple of int
        The size of the search window (in pixels) used at each pyramid level in Lucas-Kanade sparse optical flow.

    max_pyr_lvl: int
        The maximum number of pyramid levels used in Lucas Kanade sparse optical flow. As you increase this parameter larger motions can be 
        detected but consequently computation time increases.
    
    pyr_scale: float
        A float in the range [0,1] representing the downscale of the image at each pyramid level in Farneback's dense optical flow algorithm.
        For example, with a pyr_scale of 0.5, at each pyramid level the image will be half the size of the previous image.
    
    max_lk_iter: int
        The maximum number of iterations (over each frame) the Lucas-Kanade sparse optical flow algorithm will make before terminating.

    lk_accuracy_thresh: float
        A float in the range [0,1] representing the optimal termination accuracy for the Lucas-Kanade sparse optical flow algorithm.
    
    poly_sigma: float
        A floating point value representing the standard deviation of the Gaussian distribution used in the polynomial expansion of Farneback's
        dense optical flow algorithm. Typically with block_sizes of 5 or 7, a poly_sigma of 1.2 or 1.5 are used respectively.

    point_color: tuple of int
        A BGR color code of integers representing the color of points drawn over the output video. 
    
    point_radius: int
        The radius (in pixels) of points drawn over the output video.
    
    vector_color: tuple of int
        A BGR color code of integers representing the color of flow vectors drawn over the output video. If no color is provided (the default), 
        vector colors will be computed randomly.

    with_sub_dirs: bool
        Indicates whether the input directory contains subfolders.

    csv_sample_freq: int
        The time delay in milliseconds between successive csv write calls. Increase this value to speed up computation time, and decrease 
        the value to increase the number of optical flow vector samples written to the output csv file.
    
    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.

    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
    '''

    single_file = False

    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Get_optical_flow: parameter input_dir expects a string.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path in your current working tree.")
        raise OSError("Get_optical_flow: parameter input_dir is required to be a valid pathstring.")
    if os.path.isfile(input_dir):
        single_file = True
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Get_optical_flow: parameter output_dir expects a string.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path in your current working tree.")
        raise OSError("Get_optical_flow: parameter output_dir is required to be a valid path.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path to a directory in your current working tree.")
        raise OSError("Get_optical_flow: parameter output_dir must be a path string to a directory.")
    
    if not isinstance(optical_flow_type, int):
        if not isinstance(optical_flow_type, str):
            logger.warning("Function encountered a TypeError for input parameter optical_flow_type. "
                           "Message: invalid type for parameter optical_flow_type, expected str or int.")
            raise TypeError("Get_optical_flow: parameter optical_flow_type expects a string or integer.")
        elif str.lower(optical_flow_type) not in ["sparse", "dense"]:
            logger.warning("Function encountered a ValueError for input parameter optical_flow_type. "
                           "Message: unrecognized value for parameter optical_flow_type, see pyfame_utils for a full list of options.")
            raise ValueError("Get_optical_flow: parameter optical_flow_type must be one of 'sparse' or 'dense'.")
        else:
            if str.lower(optical_flow_type) == "sparse":
                optical_flow_type = SPARSE_OPTICAL_FLOW
            elif str.lower(optical_flow_type) == "dense":
                optical_flow_type = DENSE_OPTICAL_FLOW
    elif optical_flow_type not in [SPARSE_OPTICAL_FLOW, DENSE_OPTICAL_FLOW]:
        logger.warning("Function encountered a ValueError for input parameter optical_flow_type. "
                       "Message: unrecognized value for parameter optical_flow_type, see pyfame_utils for a full list of options.")
        raise ValueError("Get_optical_flow: parameter optical_flow_type must be one of SPARSE_OPTICAL_FLOW or DENSE_OPTICAL_FLOW.")
    
    if landmarks_to_track != None:
        if not isinstance(landmarks_to_track, list):
            logger.warning("Function encountered a TypeError for input parameter landmarks_to_track. "
                           "Message: invalid type for parameter landmarks_to_track, expected list[int].")
            raise TypeError("Get_optical_flow: parameter landmarks_to_track must be a list of integers.")
        for val in landmarks_to_track:
            if not isinstance(val, int):
                logger.warning("Function encountered a TypeError for input parameter landmarks_to_track. "
                               "Message: invalid type for parameter landmarks_to_track, expected list[int].")
                raise TypeError("Get_optical_flow: parameter landmarks_to_track must be a list of integers.")
    
    if not isinstance(max_corners, int):
        logger.warning("Function encountered a TypeError for input parameter max_corners. "
                       "Message: invalid type for parameter max_corners, expected int.")
        raise TypeError("Get_optical_flow: parameter max_corners must be an integer.")
    
    if not isinstance(corner_quality_lvl, float):
        logger.warning("Function encountered a TypeError for input parameter corner_quality_lvl. "
                       "Message: invalid type for parameter corner_quality_lvl, expected float.")
        raise TypeError("Get_optical_flow: parameter corner_quality_lvl must be a float.")
    elif corner_quality_lvl > 1.0 or corner_quality_lvl < 0.0:
        logger.warning("Function encountered a ValueError for input parameter corner_quality_lvl. "
                       "Message: corner_quality_lvl must be a float in the range [0,1].")
        raise ValueError("Get_optical_flow: parameter corner_quality_lvl must be a float in the range [0,1].")
    
    if not isinstance(min_corner_distance, int):
        logger.warning("Function encountered a TypeError for input parameter min_corner_distance. "
                       "Message: invalid type for parameter min_corner_distance, expected int.")
        raise TypeError("Get_optical_flow: parameter min_corner_distance must be an integer.")
    
    if not isinstance(block_size, int):
        logger.warning("Function encountered a TypeError for input parameter block_size. "
                       "Message: invalid type for parameter block_size, expected int.")
        raise TypeError("Get_optical_flow: parameter block_size must be an integer.")
    
    if not isinstance(win_size, tuple):
        logger.warning("Function encountered a TypeError for input parameter win_size. "
                        "Message: invalid type for parameter win_size, expected tuple[int].")
        raise TypeError("Get_optical_flow: parameter win_size must be a tuple of integers.")
    elif not isinstance(win_size[0], int) or not isinstance(win_size[1], int):
        logger.warning("Function encountered a TypeError for input parameter win_size. "
                       "Message: invalid type for parameter win_size, expected tuple[int].")
        raise TypeError("Get_optical_flow: parameter win_size must be a tuple of integers.")
    
    if not isinstance(max_pyr_lvl, int):
        logger.warning("Function encountered a TypeError for input parameter max_pyr_lvl. "
                       "Message: invalid type for parameter max_pyr_lvl, expected int.")
        raise TypeError("Get_optical_flow: parameter max_pyr_lvl must be an integer.")
    
    if not isinstance(pyr_scale, float):
        logger.warning("Function encountered a TypeError for input parameter pyr_scale. "
                       "Message: invalid type for parameter pyr_scale, expected float.")
        raise TypeError("Get_optical_flow: parameter pyr_scale must be a float.")
    elif pyr_scale >= 1.0 or pyr_scale < 0.0:
        logger.warning("Function encountered a ValueError for input parameter pyr_scale. "
                       "Message: pyr_scale must be a float in the range [0,1].")
        raise ValueError("Get_optical_flow: parameter pyr_scale must be a float in the range [0,1).")
    
    if not isinstance(max_lk_iter, int):
        logger.warning("Function encountered a TypeError for input parameter max_lk_iter. "
                       "Message: invalid type for parameter max_lk_iter, expected int.")
        raise TypeError("Get_optical_flow: parameter max_lk_iter must be an integer.")
    
    if not isinstance(lk_accuracy_thresh, float):
        logger.warning("Function encountered a TypeError for input parameter lk_accuracy_thresh. "
                       "Message: invalid type for parameter lk_accuracy_thresh, expected float.")
        raise TypeError("Get_optical_flow: parameter lk_accuracy_thresh must be a float.")
    elif lk_accuracy_thresh > 1.0 or lk_accuracy_thresh < 0.0:
        logger.warning("Function encountered a ValueError for input parameter lk_accuracy_thresh. "
                       "Message: lk_accuracy_thresh must be a float in the range [0,1].")
        raise ValueError("Get_optical_flow: parameter lk_accuracy_thresh must be a float in the range [0,1].")
    
    if not isinstance(poly_sigma, float):
        logger.warning("Function encountered a TypeError for input parameter poly_sigma. "
                       "Message: invalid type for parameter poly_sigma, expected float.")
        raise TypeError("Get_optical_flow: parameter poly_sigma must be a float.")
    
    if not isinstance(point_color, tuple):
        logger.warning("Function encountered a TypeError for input parameter point_color. "
                       "Message: invalid type for parameter point_color, expected tuple[int].")
        raise TypeError("Get_optical_flow: parameter point_color must be a tuple of integers.")
    for val in point_color:
        if not isinstance(val, int):
            logger.warning("Function encountered a TypeError for input parameter point_color. "
                           "Message: invalid type for parameter point_color, expected tuple[int].")
            raise TypeError("Get_optical_flow: parameter point color must be a tuple of integers.")
    
    if not isinstance(point_radius, int):
        logger.warning("Function encountered a TypeError for input parameter point_radius. "
                       "Message: invalid type for parameter point_radius, expected int.")
        raise TypeError("Get_optical_flow: parameter point_radius must be an integer.")

    if vector_color != None:
        if not isinstance(vector_color, tuple):
            logger.warning("Function encountered a TypeError for input parameter vector_color. "
                           "Message: invalid type for parameter vector_color, expected tuple[int].")
            raise TypeError("Get_optical_flow: parameter vector_color must be a tuple of integers.")
        for val in vector_color:
            if not isinstance(val, int):
                logger.warning("Function encountered a TypeError for input parameter vector_color. "
                               "Message: invalid type for parameter vector_color, expected tuple[int].")
                raise ValueError("Get_optical_flow: parameter vector_color must be a tuple of integers.")
    
    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Get_optical_flow: parameter with_sub_dirs must be a boolean.")
    
    if not isinstance(min_detection_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_detection_confidence. "
                       "Message: invalid type for parameter min_detection_confidence, expected float.")
        raise TypeError("Get_optical_flow: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_detection_confidence. "
                       "Message: min_detection_confidence must be a float in the range [0,1].")
        raise ValueError("Get_optical_flow: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_tracking_confidence. "
                       "Message: invalid type for parameter min_tracking_confidence, expected float.")
        raise TypeError("Get_optical_flow: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_tracking_confidence. "
                       "Message: min_tracking_confidence must be a float in the range [0,1].")
        raise ValueError("Get_optical_flow: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Logging input parameters
    logger.info("Now entering function get_optical_flow().")
    oft_name = None

    if isinstance(optical_flow_type, str):
        oft_name = optical_flow_type
    else:
        oft_name = get_variable_name(optical_flow_type, globals())
    
    logger.info(f"Input parameters: optical_flow_type = {oft_name}, landmarks_to_track = {landmarks_to_track}, max_corners = {max_corners}, "
                f"corner_quality_lvl = {corner_quality_lvl}, min_corner_distance = {min_corner_distance}, block_size = {block_size}, win_size = {win_size}")
    logger.info(f"Input parameters continued... max_pyr_level = {max_pyr_lvl}, pyr_scale = {pyr_scale}, max_lk_iter = {max_lk_iter}, lk_accuracy_thresh = {lk_accuracy_thresh}, "
                f"poly_sigma = {poly_sigma}, point_color = {point_color}, point_radius = {point_radius}, vector_color = {vector_color}, with_sub_dirs = {with_sub_dirs}.")
    logger.info(f"Mediapipe configurations: min_detection_confidence = {min_detection_confidence}, min_tracking_confidence = {min_tracking_confidence}.")
    
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = []

    if single_file:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Optical_Flow"):
        os.mkdir(output_dir + "\\Optical_Flow")
        output_dir = output_dir + "\\Optical_Flow"
        logger.info(f"Function created new output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\Optical_Flow"

    for file in files_to_process:

        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        csv = None
        dir_file_path = output_dir + f"\\{filename}_optical_flow{extension}"

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case _:
                logger.error("Function has encountered an unparseable file type, Function exiting with status 1. " 
                             "Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                   "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                sys.exit(1)
        
        capture = cv.VideoCapture(file)
        if not capture.isOpened():
            logger.error("Function has encountered an error attempting to initialize cv2.VideoCapture() object. "
                         "Function exiting with status 1.")
            debug_logger.error("Function has encounterd an error attempting to initialize cv2.VideoCapture() object "
                               f"with file {file}. The file may be corrupt or encoded into an unparseable file type.")
            sys.exit(1)
        
        size = (int(capture.get(3)), int(capture.get(4)))

        result = cv.VideoWriter(output_dir + "\\" + filename + "_optical_flow" + extension,
                                cv.VideoWriter.fourcc(*codec), 30, size)
        if not result.isOpened():
            logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                         "Function exiting with status 1.")
            debug_logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                               f"Please ensure that {dir_file_path} is a valid path in your current working directory tree.")
            sys.exit(1)
        
        if optical_flow_type == SPARSE_OPTICAL_FLOW:
            csv = open(output_dir + "\\" + filename + "_optical_flow.csv", "w")
            csv.write("Timestamp,X_old,Y_old,X_new,Y_new,Magnitude,Angle,Status,Error\n")
        elif optical_flow_type == DENSE_OPTICAL_FLOW:
            csv = open(output_dir + "\\" + filename + "_optical_flow.csv", "w")
            csv.write("Timestamp,dx,dy,Magnitude,Angle\n")

        
        counter = 0
        init_points = None
        mask = None
        hsv = None
        bgr = None
        old_gray = None
        rolling_time_win = csv_sample_freq
        output_img = None
        colors = np.random.randint(0,255,(max_corners,3))

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize  = win_size,
            maxLevel = max_pyr_lvl,
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, max_lk_iter, lk_accuracy_thresh))

        # Main Processing loop
        while True:
            counter += 1
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
                continue

            fo_screen_coords = []

            # Face oval screen coordinates
            for cur_source, cur_target in FACE_OVAL_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                fo_screen_coords.append((source.get('x'), source.get('y')))
                fo_screen_coords.append((target.get('x'), target.get('y')))
            
            # Create face oval image mask
            fo_mask = np.zeros((frame.shape[0], frame.shape[1]))
            fo_mask = cv.fillConvexPoly(fo_mask, np.array(fo_screen_coords), 1)
            fo_mask = fo_mask.astype(bool)

            face_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype = np.uint8)
            face_mask[fo_mask] = 255

            if counter == 1:
                if optical_flow_type == SPARSE_OPTICAL_FLOW:
                    mask = np.zeros_like(frame)
                    # Get initial tracking points
                    old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                    # If landmarks were provided 
                    if landmarks_to_track is not None:
                        init_points = []
                        init_points = np.array([[lm.get('x'), lm.get('y')] for lm in landmark_screen_coords if lm.get('id') in landmarks_to_track], dtype=np.float32)
                        init_points = init_points.reshape(-1,1,2)
                    else:
                        init_points = cv.goodFeaturesToTrack(old_gray, max_corners, corner_quality_lvl, min_corner_distance, block_size, mask=face_mask)
                elif optical_flow_type == DENSE_OPTICAL_FLOW:
                    hsv = np.zeros_like(frame)
                    old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    hsv[...,1] = 255
                
            if counter > 1:
                gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                timestamp = capture.get(cv.CAP_PROP_POS_MSEC)

                if optical_flow_type == SPARSE_OPTICAL_FLOW:

                    # Calculate optical flow
                    cur_points, st, err = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, init_points, None, **lk_params)

                    # Select good points
                    good_new_points = None
                    good_old_points = None
                    if cur_points is not None:
                        good_new_points = cur_points[st==1]
                        good_old_points = init_points[st==1]
                    
                    # Draw optical flow vectors and write out values
                    for i, (new, old) in enumerate(zip(good_new_points, good_old_points)):
                        x0, y0 = old.ravel()
                        x1, y1 = new.ravel()
                        dx = x1 - x0
                        dy = y1 - y0

                        magnitude = np.sqrt(dx**2 + dy**2)
                        angle = np.arctan2(dy, dx)

                        if timestamp > rolling_time_win:
                            # Write values to csv
                            csv.write(f"{timestamp/1000:.5f},{x0:.5f},{y0:.5f},{x1:.5f},{y1:.5f},{magnitude:.5f},{angle:.5f},{st[i][0]},{err[i][0]:.5f}\n")
                            rolling_time_win += csv_sample_freq

                        # Draw optical flow vectors on output frame
                        if vector_color == None:
                            mask = cv.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), colors[i].tolist(), 2)
                        else:
                            mask = cv.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), vector_color, 2)

                        frame = cv.circle(frame, (int(x1), int(y1)), point_radius, point_color, -1)

                    output_img = cv.add(frame, mask)
                    result.write(output_img)

                    # Update previous frame and points
                    old_gray = gray_frame.copy()
                    init_points = good_new_points.reshape(-1, 1, 2)
                
                elif optical_flow_type == DENSE_OPTICAL_FLOW:

                    # Calculate dense optical flow
                    flow = cv.calcOpticalFlowFarneback(old_gray, gray_frame, None, pyr_scale, max_pyr_lvl, win_size[0], max_lk_iter, block_size, poly_sigma, 0)

                    # Get vector magnitudes and angles
                    magnitudes, angles = cv.cartToPolar(flow[...,0],flow[...,1])

                    if timestamp > rolling_time_win:
                        for i in range(int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))):
                            for j in range(int(capture.get(cv.CAP_PROP_FRAME_WIDTH))):
                                dx = flow[i,j,0]
                                dy = flow[i,j,1]
                                csv.write(f'{timestamp/1000:.5f},{dx:.5f},{dy:.5f},{magnitudes[i,j]:.5f},{angles[i,j]:.5f}\n')
                        rolling_time_win += csv_sample_freq

                    hsv[...,0] = angles * (180/(np.pi/2))
                    hsv[...,2] = cv.normalize(magnitudes, None, 0, 255, cv.NORM_MINMAX)

                    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                    result.write(bgr)

                    old_gray = gray_frame.copy()

        logger.info(f"Function execution completed successfully, view outputted file(s) at {output_dir}.")
        capture.release()
        result.release()
        csv.close()
            
def extract_face_color_means(input_dir:str, output_dir:str, color_space: int|str = COLOR_SPACE_RGB, with_sub_dirs:bool = False,
                             min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    """Takes an input video file, and extracts colour channel means in the specified color space for the full-face, cheeks, nose and chin.
    Creates a new directory 'CSV_Output', where a csv file will be written to for each input video file provided.

    Parameters
    ----------

    input_dir: str
        A path string to a directory containing the video files to be processed.

    output_dir: str
        A path string to a directory where outputted csv files will be written to.
    
    color_space: int, str
        A specifier for which color space to operate in. One of COLOR_SPACE_RGB, COLOR_SPACE_HSV or COLOR_SPACE_GRAYSCALE
    
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

    TypeError
        Given invalid parameter types.
    ValueError 
        Given an unrecognized color space.
    OSError 
        If input or output directories are invalid paths.
    """
    
    # Global declarations and init
    singleFile = False
    static_image_mode = False

    # Type and value checking input parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Extract_color_channel_means: input_dir must be a path string.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path in your current working tree.")
        raise OSError("Extract_color_channel_means: input_dir is not a valid path.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Extract_color_channel_means: output_dir must be a path string.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path in your current working tree.")
        raise OSError("Extract_color_channel_means: output_dir is not a valid path.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path to a directory in your current working tree.")
        raise OSError("Extract_color_channel_means: output_dir must be a path string to a directory.")
    
    if isinstance(color_space, int):
        if color_space not in COLOR_SPACES:
            logger.warning("Function encountered a ValueError for input parameter color_space. "
                           "Message: unrecognized value for parameter color_space, see pyfame_utils.COLOR_SPACES for a full list of options.")
            raise ValueError("Extract_color_channel_means: unrecognized value for parameter color_space.")
    elif isinstance(color_space, str):
        if str.lower(color_space) not in ["rgb", "hsv", "grayscale"]:
            logger.warning("Function encountered a ValueError for input parameter color_space. "
                           "Message: unrecognized value for parameter color_space, see pyfame_utils.COLOR_SPACES for a full list of options.")
            raise ValueError("Extract_color_channel_means: unrecognized value for parameter color_space.")
        else:
            if str.lower(color_space) == "rgb":
                color_space = COLOR_SPACE_RGB
            if str.lower(color_space) == "hsv":
                color_space = COLOR_SPACE_HSV
            if str.lower(color_space) == "grayscale":
                color_space = COLOR_SPACE_GRAYSCALE
    else:
        logger.warning("Function encountered a TypeError for input parameter color_space. "
                       "Message: invalid type for parameter color_space, expected int or str.")
        raise TypeError("Extract_color_channel_means: color_space must be an int or str.")
    
    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Extract_color_channel_means: with_sub_dirs must be a boolean.")
    
    if not isinstance(min_detection_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_detection_confidence. "
                       "Message: invalid type for parameter min_detection_confidence, expected float.")
        raise TypeError("Extract_color_channel_means: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_detection_confidence. "
                       "Message: min_detection_confidence must be a float in the range [0,1].")
        raise ValueError("Extract_color_channel_means: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_tracking_confidence. "
                       "Message: invalid type for parameter min_tracking_confidence, expected float.")
        raise TypeError("Extract_color_channel_means: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_tracking_confidence. "
                       "Message: min_tracking_confidence must be a float in the range [0,1].")
        raise ValueError("Extract_color_channel_means: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Logging input parameters
    logger.info("Now entering function extract_face_color_means().")
    
    if isinstance(color_space, str):
        logger.info(f"Input parameters: color_space: {color_space}, with_sub_dirs = {with_sub_dirs}.")
    else:
        cs_name = get_variable_name(color_space, globals())
        logger.info(f"Input parameters: color_space: {cs_name}, with_sub_dirs = {with_sub_dirs}.")
    
    logger.info(f"Mediapipe configurations: min_detection_confidence = {min_detection_confidence}, min_tracking_confidence = {min_tracking_confidence}.")
    
    # Defining mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
    files_to_process = []

    # Creating a list of file path strings
    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")
    
    # Create an output directory for the csv files
    if not os.path.isdir(output_dir + "\\Color_Channel_Means"):
        os.mkdir(output_dir + "\\Color_Channel_Means")
        output_dir = output_dir + "\\Color_Channel_Means"
        logger.info(f"Function created a new output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\Color_Channel_Means"
    
    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        capture = None
        csv = None
        dir_file_path = output_dir

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".png" | ".jpg" | ".jpeg" | ".bmp":
                static_image_mode = True
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = True, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case _:
                logger.error("Function has encountered an unparseable file type, Function exiting with status 1. " 
                             "Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                   "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                sys.exit(1)

        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv.VideoCapture() object, exiting with status 1.")
                debug_logger.error("Function has enountered an error attempting to initialize cv.VideoCapture() object "
                                   f"with file {file}. File may be corrupt or incorrectly encoded.")
                sys.exit(1)
            
            # Writing the column headers to csv
            if color_space == COLOR_SPACE_RGB:
                dir_file_path += f"\\{filename}_RGB.csv"
                csv = open(output_dir + "\\" + filename + "_RGB.csv", "w")
                csv.write("Timestamp,Mean_Red,Mean_Green,Mean_Blue,Cheeks_Red,Cheeks_Green,Cheeks_Blue," +
                        "Nose_Red,Nose_Green,Nose_Blue,Chin_Red,Chin_Green,Chin_Blue\n")
            elif color_space == COLOR_SPACE_HSV:
                dir_file_path += f"\\{filename}_HSV.csv"
                csv = open(output_dir + "\\" + filename + "_HSV.csv", "w")
                csv.write("Timestamp,Mean_Hue,Mean_Sat,Mean_Value,Cheeks_Hue,Cheeks_Sat,Cheeks_Value," + 
                        "Nose_Hue,Nose_Sat,Nose_Value,Chin_Hue,Chin_Sat,Chin_Value\n")
            elif color_space == COLOR_SPACE_GRAYSCALE:
                dir_file_path += f"\\{filename}_GRAYSCALE.csv"
                csv = open(output_dir + "\\" + filename + "_GRAYSCALE.csv", "w")
                csv.write("Timestamp,Mean_Value,Cheeks_Value,Nose_Value,Chin_Value\n")
        else:
            # Writing the column headers to csv
            if color_space == COLOR_SPACE_RGB:
                dir_file_path += f"\\{filename}_RGB.csv"
                csv = open(output_dir + "\\" + filename + "_RGB.csv", "w")
                csv.write("Mean_Red,Mean_Green,Mean_Blue,Cheeks_Red,Cheeks_Green,Cheeks_Blue," +
                        "Nose_Red,Nose_Green,Nose_Blue,Chin_Red,Chin_Green,Chin_Blue\n")
            elif color_space == COLOR_SPACE_HSV:
                dir_file_path += f"\\{filename}_HSV.csv"
                csv = open(output_dir + "\\" + filename + "_HSV.csv", "w")
                csv.write("Mean_Hue,Mean_Sat,Mean_Value,Cheeks_Hue,Cheeks_Sat,Cheeks_Value," + 
                        "Nose_Hue,Nose_Sat,Nose_Value,Chin_Hue,Chin_Sat,Chin_Value\n")
            elif color_space == COLOR_SPACE_GRAYSCALE:
                dir_file_path += f"\\{filename}_GRAYSCALE.csv"
                csv = open(output_dir + "\\" + filename + "_GRAYSCALE.csv", "w")
                csv.write("Mean_Value,Cheeks_Value,Nose_Value,Chin_Value\n")
    
    while True:
        if not static_image_mode:
            success, frame = capture.read()
            if not success:
                break
        else:
            frame = cv.imread(file)
            if frame is None:
                logger.error("Function has encountered an error attempting to call cv.imread(), exiting with status 1.")
                debug_logger.error("Function has encountered an error attempting to call cv.imread() with filepath "
                                   f"{file}. File may be corrupt or incorrectly encoded.")
                sys.exit(1)

        face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        landmark_screen_coords = []

        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:

                # Convert normalised landmark coordinates to x-y pixel coordinates
                for id,lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = frame.shape
                    x,y = int(lm.x * iw), int(lm.y * ih)
                    landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
        elif static_image_mode:
            debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process().")
            logger.error("Face mesh detection error, function exiting with status 1.")
            sys.exit(1)
        else: 
            continue
        
        # Concave Polygons
        lc_screen_coords = []
        rc_screen_coords = []
        chin_screen_coords = []

        lc_path = create_path(LEFT_CHEEK_IDX)
        rc_path = create_path(RIGHT_CHEEK_IDX)
        chin_path = create_path(CHIN_IDX)

        # Left cheek screen coordinates
        for cur_source, cur_target in lc_path:
            source = landmark_screen_coords[cur_source]
            target = landmark_screen_coords[cur_target]
            lc_screen_coords.append((source.get('x'),source.get('y')))
            lc_screen_coords.append((target.get('x'),target.get('y')))

        # Right cheek screen coordinates
        for cur_source, cur_target in rc_path:
            source = landmark_screen_coords[cur_source]
            target = landmark_screen_coords[cur_target]
            rc_screen_coords.append((source.get('x'),source.get('y')))
            rc_screen_coords.append((target.get('x'),target.get('y')))
        
        # Chin screen coordinates
        for cur_source, cur_target in chin_path:
            source = landmark_screen_coords[cur_source]
            target = landmark_screen_coords[cur_target]
            chin_screen_coords.append((source.get('x'),source.get('y')))
            chin_screen_coords.append((target.get('x'),target.get('y')))

        lc_screen_coords = np.array(lc_screen_coords, dtype=np.int32)
        rc_screen_coords = np.array(rc_screen_coords, dtype=np.int32)
        chin_screen_coords = np.array(chin_screen_coords, dtype=np.int32)

        lc_screen_coords.reshape((-1, 1, 2))
        rc_screen_coords.reshape((-1, 1, 2))
        chin_screen_coords.reshape((-1, 1, 2))

        # Creating concave polygon masks
        lc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        lc_mask = cv.fillPoly(img=lc_mask, pts=[lc_screen_coords], color=(255,255,255))
        lc_mask = lc_mask.astype(bool)

        rc_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        rc_mask = cv.fillPoly(img=rc_mask, pts=[rc_screen_coords], color=(255,255,255))
        rc_mask = rc_mask.astype(bool)

        chin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        chin_mask = cv.fillPoly(img=chin_mask, pts=[chin_screen_coords], color=(255,255,255))
        chin_mask = chin_mask.astype(bool)

        # Convex polygons
        nose_screen_coords = []
        le_screen_coords = []
        re_screen_coords = []
        lips_screen_coords = []
        face_oval_screen_coords = []

        # Face oval screen coordinates
        for cur_source, cur_target in FACE_OVAL_TIGHT_PATH:
            source = landmark_screen_coords[cur_source]
            target = landmark_screen_coords[cur_target]
            face_oval_screen_coords.append((source.get('x'),source.get('y')))
            face_oval_screen_coords.append((target.get('x'),target.get('y')))
        
        # Left Eye screen coordinates
        for cur_source, cur_target in LEFT_EYE_PATH:
            source = landmark_screen_coords[cur_source]
            target = landmark_screen_coords[cur_target]
            le_screen_coords.append((source.get('x'),source.get('y')))
            le_screen_coords.append((target.get('x'),target.get('y')))
        
        # Right Eye screen coordinates
        for cur_source, cur_target in RIGHT_EYE_PATH:
            source = landmark_screen_coords[cur_source]
            target = landmark_screen_coords[cur_target]
            re_screen_coords.append((source.get('x'),source.get('y')))
            re_screen_coords.append((target.get('x'),target.get('y')))
        
        # Right Eye screen coordinates
        for cur_source, cur_target in NOSE_PATH:
            source = landmark_screen_coords[cur_source]
            target = landmark_screen_coords[cur_target]
            nose_screen_coords.append((source.get('x'),source.get('y')))
            nose_screen_coords.append((target.get('x'),target.get('y')))

        # Right Eye screen coordinates
        for cur_source, cur_target in LIPS_PATH:
            source = landmark_screen_coords[cur_source]
            target = landmark_screen_coords[cur_target]
            lips_screen_coords.append((source.get('x'),source.get('y')))
            lips_screen_coords.append((target.get('x'),target.get('y')))
        
        # Use screen coordinates to create boolean mask
        face_oval_mask = np.zeros((frame.shape[0],frame.shape[1]))
        face_oval_mask = cv.fillConvexPoly(face_oval_mask, np.array(face_oval_screen_coords), 1)
        face_oval_mask = face_oval_mask.astype(bool)

        le_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
        le_mask = le_mask.astype(bool)
        
        re_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
        re_mask = re_mask.astype(bool)

        nose_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        nose_mask = cv.fillConvexPoly(nose_mask, np.array(nose_screen_coords), 1)
        nose_mask = nose_mask.astype(bool)

        lips_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        lips_mask = cv.fillConvexPoly(lips_mask, np.array(lips_screen_coords), 1)
        lips_mask = lips_mask.astype(bool)

        # Create binary image masks 
        bin_fo_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        bin_fo_mask[face_oval_mask] = 255
        bin_fo_mask[le_mask] = 0
        bin_fo_mask[le_mask] = 0
        bin_fo_mask[lips_mask] = 0

        bin_cheeks_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        bin_cheeks_mask[lc_mask] = 255
        bin_cheeks_mask[rc_mask] = 255

        bin_nose_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        bin_nose_mask[nose_mask] = 255

        bin_chin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        bin_chin_mask[chin_mask] = 255

        if not static_image_mode: 
            if color_space == COLOR_SPACE_RGB:
                # Extracting the color channel means
                blue, green, red, *_ = cv.mean(frame, bin_fo_mask)
                b_cheeks, g_cheeks, r_cheeks, *_ = cv.mean(frame, bin_cheeks_mask)
                b_nose, g_nose, r_nose, *_ = cv.mean(frame, bin_nose_mask)
                b_chin, g_chin, r_chin, *_ = cv.mean(frame, bin_chin_mask)

                # Get the current video timestamp 
                timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000

                csv.write(f"{timestamp:.5f},{red:.5f},{green:.5f},{blue:.5f}," +
                        f"{r_cheeks:.5f},{g_cheeks:.5f},{b_cheeks:.5f}," + 
                        f"{r_nose:.5f},{g_nose:.5f},{b_nose:.5f}," + 
                        f"{r_chin:.5f},{g_chin:.5f},{b_chin:.5f}\n")

            elif color_space == COLOR_SPACE_HSV:
                # Extracting the color channel means
                hue, sat, val, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_fo_mask)
                h_cheeks, s_cheeks, v_cheeks, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_cheeks_mask)
                h_nose, s_nose, v_nose, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_nose_mask)
                h_chin, s_chin, v_chin, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_chin_mask)

                # Get the current video timestamp
                timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000

                csv.write(f"{timestamp:.5f},{hue:.5f},{sat:.5f},{val:.5f}," +
                        f"{h_cheeks:.5f},{s_cheeks:.5f},{v_cheeks:.5f}," +
                        f"{h_nose:.5f},{s_nose:.5f},{v_nose:.5f}," + 
                        f"{h_chin:.5f},{s_chin:.5f},{v_chin:.5f}\n")
            
            elif color_space == COLOR_SPACE_GRAYSCALE:
                # Extracting the color channel means
                val, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_fo_mask)
                v_cheeks, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_cheeks_mask)
                v_nose, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_nose_mask)
                v_chin, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_chin_mask)

                # Get the current video timestamp
                timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000

                csv.write(f"{timestamp:.5f},{val:.5f},{v_cheeks:.5f},{v_nose:.5f},{v_chin:.5f}\n")
        else:
            if color_space == COLOR_SPACE_RGB:
                # Extracting the color channel means
                blue, green, red, *_ = cv.mean(frame, bin_fo_mask)
                b_cheeks, g_cheeks, r_cheeks, *_ = cv.mean(frame, bin_cheeks_mask)
                b_nose, g_nose, r_nose, *_ = cv.mean(frame, bin_nose_mask)
                b_chin, g_chin, r_chin, *_ = cv.mean(frame, bin_chin_mask)

                csv.write(f"{red:.5f},{green:.5f},{blue:.5f}," +
                        f"{r_cheeks:.5f},{g_cheeks:.5f},{b_cheeks:.5f}," + 
                        f"{r_nose:.5f},{g_nose:.5f},{b_nose:.5f}," + 
                        f"{r_chin:.5f},{g_chin:.5f},{b_chin:.5f}\n")

            elif color_space == COLOR_SPACE_HSV:
                # Extracting the color channel means
                hue, sat, val, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_fo_mask)
                h_cheeks, s_cheeks, v_cheeks, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_cheeks_mask)
                h_nose, s_nose, v_nose, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_nose_mask)
                h_chin, s_chin, v_chin, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_chin_mask)

                csv.write(f"{hue:.5f},{sat:.5f},{val:.5f}," +
                        f"{h_cheeks:.5f},{s_cheeks:.5f},{v_cheeks:.5f}," +
                        f"{h_nose:.5f},{s_nose:.5f},{v_nose:.5f}," + 
                        f"{h_chin:.5f},{s_chin:.5f},{v_chin:.5f}\n")
            
            elif color_space == COLOR_SPACE_GRAYSCALE:
                # Extracting the color channel means
                val, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_fo_mask)
                v_cheeks, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_cheeks_mask)
                v_nose, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_nose_mask)
                v_chin, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_chin_mask)

                csv.write(f"{val:.5f},{v_cheeks:.5f},{v_nose:.5f},{v_chin:.5f}\n")
            
            break
    
    if not static_image_mode:
        capture.release()
    csv.close()
    logger.info(f"Function execution completed successfully, view outputted file(s) at {dir_file_path}.")

def generate_shuffled_block_array(file_path:str, shuffle_method:int = FRAME_SHUFFLE_RANDOM, rand_seed:int|None = None, block_duration:int = 1000) -> tuple:
    """ This function takes a given file path and shuffling method, and returns a tuple containing the block_order array and the block size parameter used
    in shuffle_frame_order(). The output of generate_shuffled_block_array can be directly fed into shuffle_frame_order as the block_order parameter.

    Parameters
    ----------

    file_path: str
        A path string to a video file in your current working directory tree.

    shuffle_method: int
        An integer flag indicating the shuffling method used. For a full list of options please see pyfame_utils.

    rand_seed: int | None
        A seed to numpy's random number generator.
    
    block_duration: int
        The precise time duration in milliseconds of each block of frames.

    Raises
    ------

    OSError: given invalid input file paths.
    TypeError: given invalid parameter types.
    ValueError: given unrecognized values for input parameters.
    """ 

    logger.info("Now entering function generate_shuffled_block_array().")

    # Input parameter checks
    if not isinstance(file_path, str):
        logger.warning("Function encountered a TypeError for input parameter file_path. "
                       "Message: invalid type for parameter file_path, expected str.")
        raise TypeError("Generate_shuffled_block_array: invalid type for parameter file_path.")
    elif not os.path.exists(file_path):
        logger.warning("Function encountered an OSError for input parameter file_path. "
                       "Message: file_path is not a valid path in your current working tree.")
        raise OSError("Generate_shuffled_block_array: file_path is not a valid path, or path does not exist.")
    elif not os.path.isfile(file_path):
        logger.warning("Function encountered a ValueError for input parameter file_path. "
                       "Message: file_path is not a valid path to a file in your current working tree.")
        raise ValueError("Generate_shuffled_block_array: file_path must be a valid path to a video file.")
    
    if not isinstance(shuffle_method, int):
        logger.warning("Function encountered a TypeError for input parameter shuffle_method. "
                       "Message: invalid type for parameter shuffle_method, expected int.")
        raise TypeError("Generate_shuffled_block_array: parameter running_mode must be an integer.")
    # add valueError check

    if rand_seed != None:
        if not isinstance(rand_seed, int):
            logger.warning("Function encountered a TypeError for input parameter rand_seed. "
                       "Message: invalid type for parameter rand_seed, expected int.")
            raise TypeError("Generate_shuffled_block_array: parameter rand_seed must be an integer.")
    
    if not isinstance(block_duration, int):
        logger.warning("Function encountered a TypeError for input parameter block_duration. "
                       "Message: invalid type for parameter block_duration, expected int.")
        raise TypeError("Generate_shuffled_block_array: parameter block_size must be an integer")
    
    # Logging input parameters
    shuffle_method_name = get_variable_name(shuffle_method, globals())
    logger.info(f"Input Parameters: file_path = {file_path}, shuffle_method = {shuffle_method_name}, rand_seed = {rand_seed}, "
                f"block_duration = {block_duration}.")

    # Initialize videocapture object
    capture = cv.VideoCapture(file_path)
    if not capture.isOpened():
        logger.error("Function has encountered an error attempting to initialize cv.VideoCapture() object, exiting with status 1.")
        debug_logger.error("Function has enountered an error attempting to initialize cv.VideoCapture() object "
                           f"with file {file_path}. File may be corrupt or incorrectly encoded.")
        sys.exit(1)
    
    # Initialize numpy random number generator
    rng = None
    if rand_seed != None:
        rng = np.random.default_rng(seed=rand_seed)
    else:
        rng = np.random.default_rng()

    # Retrieve relevant capture properties to compute block_size
    num_frames = capture.get(cv.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv.CAP_PROP_FPS)
    video_duration = (num_frames/fps) * 1000
    num_blocks = (video_duration / block_duration)
    block_size = int(num_frames//num_blocks)

    num_blocks = int(np.ceil(num_blocks))
    block_order = []

    for i in range(num_blocks):
        block_order.append(i)

    match shuffle_method:
        case 34:
            rng.shuffle(block_order)

        case 35:
            block_order = rng.choice(block_order, size=num_blocks, replace=True)
        
        case 36:
            block_order.reverse()

    logger.info(f"Function execution completed, returning ({block_order}, {block_size}).")
    return (block_order, block_size)

def shuffle_frame_order(input_dir:str, output_dir:str, shuffle_method:int = FRAME_SHUFFLE_RANDOM, rand_seed:int|None = None, block_order:tuple|None = None, 
                        block_duration:int = 1000, drop_last_block:bool = True, with_sub_dirs:bool = False) -> None:
    """For each video file contained within input_dir, randomly shuffles the frame order by shuffling blocks of frames. Use utility function generate_shuffled_block_array()
    to pre-generate the block_order list. The output of generate_shuffled_block_array() can be directly passed as the value of input parameter block_order. Alternatively, 
    simply specify the shuffle_method of choice and the block_duration, and shuffle_frame_order() will invoke generate_shuffled_block_array() internally. After shuffling the 
    block order, the function writes the processed file to output_dir. 
    
    Parameters
    ----------
    input_dir: str
        A path string to the directory containing input video files.

    output_dir: str
        A path string to the directory where outputted video files will be saved.

    shuffle_method: int
        An integer flag indicating the functions running mode. One of SHUFFLE_FRAME_ORDER or REVERSE_FRAME_ORDER.
    
    rand_seed: int
        The seed number provided to the numpy random generator instance.
    
    block_order: tuple or None
        A tuple (as output of generate_shuffled_block_array()) containing a (list, int), where the list is the block ordering
        and the integer specifies the number of frames per block. If None, the tuple contents will be computed internally.

    block_duration: int
        The time duration (in milliseconds) of each block of frames.

    drop_last_block: bool
        A boolean flag indicating if the uneven block of remaining frames should be dropped from the output.
    
    with_sub_dirs: bool
        A boolean flag indicating if the input directory contains subdirectories.
    
    Raises
    ------

    TypeError: given invalid parameter types.
    OSError: given invalid file paths to input_dir or output_dir.
    """

    logger.info("Now entering function shuffle_frame_order().")

    single_file = False
    rng = None

    # Performing checks on function parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Shuffle_frame_order: invalid type for parameter input_dir.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path in your current working tree.")
        raise OSError("Shuffle_frame_order: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        single_file = True
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Shuffle_frame_order: parameter output_dir must be a str.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path in your current working tree.")
        raise OSError("Shuffle_frame_order: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path to a directory in your current working tree.")
        raise ValueError("Shuffle_frame_order: output_dir must be a valid path to a directory.")
    
    if not isinstance(shuffle_method, int):
        logger.warning("Function encountered a TypeError for input parameter shuffle_method. "
                       "Message: invalid type for parameter shuffle_method, expected int.")
        raise TypeError("Shuffle_frame_order: parameter running_mode must be an integer.")
    # add valueerror check

    if rand_seed != None:
        if not isinstance(rand_seed, int):
            logger.warning("Function encountered a TypeError for input parameter rand_seed. "
                       "Message: invalid type for parameter rand_seed, expected int.")
            raise TypeError("Shuffle_frame_order: parameter rand_seed must be an integer.")
    
    if not isinstance(block_duration, int):
        logger.warning("Function encountered a TypeError for input parameter block_duration. "
                       "Message: invalid type for parameter block_duration, expected int.")
        raise TypeError("Shuffle_frame_order: parameter block_size must be an integer")
    
    if block_order != None:
        if not isinstance(block_order, list):
            logger.warning("Function encountered a TypeError for input parameter block_order. "
                           "Message: invalid type for parameter block_order, expected list.")
            raise TypeError("Shuffle_frame_order: parameter block_order must be a zero-indexed list of integers.")
        
        for i in block_order:
            if not isinstance(i, int):
                logger.warning("Function encountered a TypeError for input parameter block_order. "
                               "Message: invalid type for parameter block_order, expected list[int].")
                raise TypeError("Shuffle_frame_order: parameter block_order must be a zero-indexed list of integers.")
    
    if not isinstance(drop_last_block, bool):
        logger.warning("Function Encountered a TypeError for input parameter drop_last_block. "
                       "Message: invalid type for parameter drop_last_block, expected bool.")
        raise TypeError("Shuffle_frame_order: parameter drop_last_block must be a bool.")

    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function Encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Shuffle_frame_order: parameter with_sub_dirs must be a bool.")
    
    # Logging input parameters
    shuffle_method_name = get_variable_name(shuffle_method, globals())
    logger.info(f"Input Parameters: shuffle_method = {shuffle_method_name}, rand_seed = {rand_seed}, block_order = {block_order}, "
                f"block_duration = {block_duration}, drop_last_block = {drop_last_block}, with_sub_dirs = {with_sub_dirs}.")
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = []

    if single_file:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
        
    logger.info(f"Function has read in {len(files_to_process)} files from input directory {input_dir}.")
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Frame_Shuffle"):
        os.mkdir(output_dir + "\\Frame_Shuffle")
        output_dir = output_dir + "\\Frame_Shuffle"
        logger.info(f"Function created new output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\Frame_Shuffle"

    if rand_seed == None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(rand_seed)

    for file in files_to_process:
            
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        block_size = None
        dir_file_path = output_dir + f"{filename}_frame_shuffled{extension}"

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
            case ".mov":
                codec = "MP4V"
            case _:
                logger.error("Function has encountered an unparseable file type, Function exiting with status 1. " 
                             "Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                   "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                sys.exit(1)
      
        capture = cv.VideoCapture(file)
        if not capture.isOpened():
            logger.error("Function has encountered an error attempting to initialize cv2.VideoCapture() object. "
                         "Function exiting with status 1.")
            debug_logger.error("Function has encounterd an error attempting to initialize cv2.VideoCapture() object "
                               f"with file {file}. The file may be corrupt or encoded into an unparseable file type.")
            sys.exit(1)
        
        size = (int(capture.get(3)), int(capture.get(4)))

        result = cv.VideoWriter(output_dir + "\\" + filename + "_frame_shuffled" + extension,
                                cv.VideoWriter.fourcc(*codec), 30, size)
        if not result.isOpened():
            logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                         "Function exiting with status 1.")
            debug_logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                               f"Please ensure that {dir_file_path} is a valid path in your current working directory tree.")
            sys.exit(1)

        if block_order is None:
            block_order, block_size = generate_shuffled_block_array(file, shuffle_method, rand_seed, block_duration)
        else:
            block_order, block_size = block_order
        
        match shuffle_method:
            case 34 | 35 | 36:
                shuffled_frames = {}
                counter = 0
                cur_block = []

                # Read in and store all frames
                while ret := capture.read():
                    success, frame = ret
                    if success:
                        cur_block.append(frame)
                        if len(cur_block) == block_size:
                            shuffled_frames.update({counter:cur_block.copy()})
                            cur_block = []
                            counter += 1
                    elif len(cur_block) > 0:
                        shuffled_frames.update({counter:cur_block.copy()})
                        break
                    else:
                        break
                
                if drop_last_block:
                    key = len(shuffled_frames) - 1 
                    shuffled_frames.pop(key)

                    largest_key_idx = block_order.index(max(block_order))
                    del block_order[largest_key_idx]

                original_keys = list(shuffled_frames.keys())
                ref_dict = shuffled_frames.copy()

                for old_key,new_key in zip(original_keys,block_order):
                    new_block = ref_dict[new_key]
                    shuffled_frames.update({old_key:new_block})
                    
                for key in original_keys:
                    block = shuffled_frames.get(key)
                    for out_frame in block:
                        result.write(out_frame)
            
            case 37:
                shuffled_frames = {}
                counter = 0
                cur_block = []

                # Read in and store all frames
                while ret := capture.read():
                    success, frame = ret
                    if success:
                        cur_block.append(frame)
                        if len(cur_block) == block_size:
                            shuffled_frames.update({counter:cur_block.copy()})
                            cur_block = []
                            counter += 1
                    elif len(cur_block) > 0:
                        shuffled_frames.update({counter:cur_block.copy()})
                        break
                    else:
                        break
                
                if drop_last_block:
                    key = len(shuffled_frames) - 1 
                    shuffled_frames.pop(key)
                
                original_keys = list(shuffled_frames.keys())

                # Perform right cyclic shift
                for key in original_keys:
                    block = shuffled_frames.get(key)
                    new_block = [block[-1]] + block[:-1]
                    shuffled_frames.update({key:new_block})
                
                for key in original_keys:
                    block = shuffled_frames.get(key)
                    for out_frame in block:
                        result.write(out_frame)
            
            case 38:
                shuffled_frames = {}
                counter = 0
                cur_block = []

                # Read in and store all frames
                while ret := capture.read():
                    success, frame = ret
                    if success:
                        cur_block.append(frame)
                        if len(cur_block) == block_size:
                            shuffled_frames.update({counter:cur_block.copy()})
                            cur_block = []
                            counter += 1
                    elif len(cur_block) > 0:
                        shuffled_frames.update({counter:cur_block.copy()})
                        break
                    else:
                        break
                
                if drop_last_block:
                    key = len(shuffled_frames) - 1 
                    shuffled_frames.pop(key)
                
                original_keys = list(shuffled_frames.keys())
                # Perform left cyclic shift
                for key in original_keys:
                    block = shuffled_frames.get(key)
                    new_block = block[1:] + [block[0]]
                    shuffled_frames.update({key:new_block})
                    
                for key in original_keys:
                    block = shuffled_frames.get(key)
                    for out_frame in block:
                        result.write(out_frame)
            
            case 39:
                shuffled_frames = {}
                counter = 0
                cur_block = []

                # Read in and store all frames
                while ret := capture.read():
                    success, frame = ret
                    if success:
                        cur_block.append(frame)
                        if len(cur_block) == block_size:
                            shuffled_frames.update({counter:cur_block.copy()})
                            cur_block = []
                            counter += 1
                    elif len(cur_block) > 0:
                        shuffled_frames.update({counter:cur_block.copy()})
                        break
                    else:
                        break
                
                if drop_last_block:
                    key = len(shuffled_frames) - 1 
                    shuffled_frames.pop(key)
                
                shuffled_palindrome_frames = {}
                original_keys = list(shuffled_frames.keys())
                counter = 0

                # Perform palindrome stutter
                for key in original_keys:
                    block = shuffled_frames.get(key)
                    rev_block = block[::-1]
                    shuffled_palindrome_frames.update({counter:block})
                    shuffled_palindrome_frames.update({counter+1:rev_block})
                    counter += 2
                
                palindrome_keys = list(shuffled_palindrome_frames.keys())
                for key in palindrome_keys:
                    block = shuffled_palindrome_frames.get(key)
                    for out_frame in block:
                        result.write(out_frame)

            case 40:
                shuffled_frames = {}
                counter = 0
                cur_block = []

                # Read in and store all frames
                while ret := capture.read():
                    success, frame = ret
                    if success:
                        cur_block.append(frame)
                        if len(cur_block) == block_size:
                            shuffled_frames.update({counter:cur_block.copy()})
                            cur_block = []
                            counter += 1
                    elif len(cur_block) > 0:
                        shuffled_frames.update({counter:cur_block.copy()})
                        break
                    else:
                        break

                shuffled_frames.pop(len(shuffled_frames)-1)

                # perform interleaving of frames
                interleaved_frames = list(itertools.chain(*zip(*(shuffled_frames.values()))))

                for frame in interleaved_frames:
                    result.write(frame)
        
        capture.release()
        result.release()
        logger.info(f"Function execution completed successfully, view outputted file(s) at {dir_file_path}.")
        
        
def face_color_shift(input_dir:str, output_dir:str, onset_t:float = 0.0, offset_t:float = 0.0, shift_magnitude: float = 8.0, timing_func:Callable[...,float] = sigmoid, 
                     shift_color:str|int = COLOR_RED, landmark_regions:list[list[tuple]] = FACE_SKIN_PATH, with_sub_dirs:bool = False, 
                     min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5, **kwargs) -> None: 
    """For each image or video file contained in input_dir, the function applies a weighted color shift to the face region, 
    outputting each resulting file in output_dir. Weights are calculated using a passed timing function, that returns
    a float in the normalised range [0,1]. Any additional keyword arguments will be passed to the specified timing function.
    (NOTE there is currently no checking to ensure timing function outputs are normalised)

    Parameters
    ----------

    input_dir: str
        A path string to the directory containing input video files.

    output_dir: str
        A path string to the directory where outputted video files will be saved.
    
    onset_t: float
        The onset time of the colour shifting.
    
    offset_t: float
        The offset time of the colour shifting.
    
    shift_magnitude: float
        The maximum units to shift the colour temperature by, during peak onset.
    
    timingFunc: Function() -> float
        Any function that takes at least one input float (time), and returns a float.

    shift_color: str, int
        Either a string literal specifying the color of choice, or a predefined integer constant.
    
    landmark_regions: list of list, list of tuple
        A list of one or more landmark paths, specifying the region in which the colouring will take place.
    
    with_sub_dirs: bool
        A boolean flag indicating whether the input directory contains nested directories.
    
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
    OSError
        Given invalid directory paths.
    ValueError:
        If provided timing_func does not return a normalised float value.
    """

    logger.info("Now entering function face_color_shift().")

    singleFile = False
    static_image_mode = False

    def shift_color_temp(img: cv2.typing.MatLike, img_mask: cv2.typing.MatLike | None, shift_weight: float, max_color_shift: float = 8.0, 
                    shift_color: str|int = COLOR_RED) -> cv2.typing.MatLike:
        """Takes in an image and a mask of the same shape, and shifts the specified color temperature by (weight * max_shift) 
        units in the masked region of the image. This function makes use of the CIE La*b* perceptually uniform color space to 
        perform natural looking color shifts on the face.

        Parameters
        ----------

        img: Matlike
            An input still image or video frame.

        img_mask: Matlike
            A binary image with the same shape as img.

        shift_weight: float
            The current shifting weight; a float in the range [0,1] returned from a timing function. 

        max_color_shift: float
            The maximum units to shift a* (red-green) or b* (blue-yellow) of the Lab* color space.
        
        shift_color: str, int
            An integer or string literal specifying which color will be applied to the input image.
                
        Raises
        ------

        TypeError
            On invalid input parameter types.
        ValueError 
            If an undefined color value is passed, or non-matching image and mask shapes are provided.

        Returns
        -------

        result: Matlike
            The input image, color-shifted in the region specified by the input mask. 
        """

        # Convert input image to CIE La*b* color space (perceptually uniform space)
        img_LAB = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)
        l,a,b = cv.split(img_LAB)

        if shift_color == COLOR_RED or str.lower(shift_color) == "red":
            a = np.where(img_mask==255, a + (shift_weight * max_color_shift), a)
            np.clip(a, -128, 127)
        if shift_color == COLOR_BLUE or str.lower(shift_color) == "blue":
            b = np.where(img_mask==255, b - (shift_weight * max_color_shift), b)
            np.clip(a, -128, 127)
        if shift_color == COLOR_GREEN or str.lower(shift_color) == "green":
            a = np.where(img_mask==255, a - (shift_weight * max_color_shift), a)
            np.clip(a, -128, 127)
        if shift_color == COLOR_YELLOW or str.lower(shift_color) == "yellow":
            b = np.where(img_mask==255, b + (shift_weight * max_color_shift), b)
            np.clip(a, -128, 127)
        
        img_LAB = cv.merge([l,a,b])
        
        # Convert CIE La*b* back to BGR
        result = cv.cvtColor(img_LAB.astype(np.uint8), cv.COLOR_LAB2BGR)
        return result
    
    # Performing checks on function parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Face_color_shift: invalid type for parameter input_dir.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path in your current working tree.")
        raise OSError("Face_color_shift: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Face_color_shift: parameter output_dir must be a str.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path in your current working tree.")
        raise OSError("Face_color_shift: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered a ValueError for input parameter output_dir. "
                       "Message: output_dir is not a valid path to a directory in your current working tree.")
        raise ValueError("Face_color_shift: output_dir must be a valid path to a directory.")
    
    if not isinstance(onset_t, float):
        logger.warning("Function encountered a TypeError for input parameter onset_t. "
                       "Message: invalid type for parameter onset_t, expected float.")
        raise TypeError("Face_color_shift: parameter onset_t must be a float.")
    if not isinstance(offset_t, float):
        logger.warning("Function encountered a TypeError for input parameter offset_t. "
                       "Message: invalid type for parameter offset_t, expected float.")
        raise TypeError("Face_color_shift: parameter offset_t must be a float.")
    if not isinstance(shift_magnitude, float):
        logger.warning("Function encountered a TypeError for input parameter shift_magnitude. "
                       "Message: invalid type for parameter shift_magnitude, expected float.")
        raise TypeError("Face_color_shift: parameter shift_magnitude must be a float.")

    if isinstance(shift_color, str):
        if str.lower(shift_color) not in ["red", "green", "blue", "yellow"]:
            logger.warning("Function encountered a ValueError for input parameter shift_color. "
                           "Message: unrecognized value for parameter shift_color, see pyfame_utils for a full list of accepted options.")
            raise ValueError("Face_color_shift: shift_color must be one of: red, green, blue, yellow.")
    elif isinstance(shift_color, int):
        if shift_color not in [COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW]:
            logger.warning("Function encountered a ValueError for input parameter shift_color. "
                           "Message: unrecognized value for parameter shift_color, see pyfame_utils for a full list of accepted options.")
            raise ValueError("Face_color_shift: shift_color must be one of: red, green, blue, yellow.")
    else:
        logger.warning("Function encountered a TypeError for input parameter shift_color. "
                       "Message: invalid type for parameter shift_color, expected int or str.")
        raise TypeError("Face_color_shift: shift_color must be of type str or int.")

    if not isinstance(landmark_regions, list):
        logger.warning("Function encountered a TypeError for input parameter landmark_regions. "
                       "Message: invalid type for parameter landmark_regions, expected list.")
        raise TypeError("Face_color_shift: parameter landmarks_to_color expects a list.")
    for val in landmark_regions:
        if not isinstance(val, list) or not isinstance(val, tuple):
            logger.warning("Function encountered a ValueError for input parameter landmark_regions. "
                           "Message: landmark_regions must either be a list[list[tuple]] or list[tuple].")
            raise ValueError("Face_color_shift: landmarks_to_color may either be a list of lists, or a singular list of tuples.")

    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Face_color_shift: parameter with_sub_dirs must be of type bool.")
    
    if not isinstance(min_detection_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_detection_confidence. "
                       "Message: invalid type for parameter min_detection_confidence, expected float.")
        raise TypeError("Face_color_shift: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_detection_confidence. "
                       "Message: min_detection_confidence must be a float in the range [0,1].")
        raise ValueError("Face_color_shift: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_tracking_confidence. "
                       "Message: invalid type for parameter min_tracking_confidence, expected float.")
        raise TypeError("Face_color_shift: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_tracking_confidence. "
                       "Message: min_tracking_confidence must be a float in the range [0,1].")
        raise ValueError("Face_color_shift: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Logging input parameters
    shift_color_name = ""
    if isinstance(shift_color, str):
        shift_color_name = shift_color
    else:
        shift_color_name = get_variable_name(shift_color, globals())

    landmark_names = "["
    for i in range(len(landmark_regions)):
        cur_name = get_variable_name(landmark_regions[i], globals())
        if i == len(landmark_regions) - 1:
            landmark_names += f"{cur_name}]"
        else:
            landmark_names += f"{cur_name}, "

    logger.info(f"Input parameters: onset_t = {onset_t}, offset_t = {offset_t}, shift_magnitude = {shift_magnitude}, "
                f"shift_color = {shift_color_name}, landmark_regions = {landmark_names}, with_sub_dirs = {with_sub_dirs}.")
    logger.info(f"Mediapipe configurations: min_detection_confidence = {min_detection_confidence}, min_tracking_confidence = {min_tracking_confidence}.")

    # Creating a list of file path strings to iterate through when processing
    files_to_process = []

    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Color_Shifted"):
        os.mkdir(output_dir + "\\Color_Shifted")
        output_dir = output_dir + "\\Color_Shifted"
        logger.info(f"Function created a new output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\Color_Shifted"
    
    for file in files_to_process:
            
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        cap_duration = None
        dir_file_path = output_dir + f"{filename}_color_shifted{extension}"

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
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
                sys.exit(1)
        
        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoCapture() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encounterd an error attempting to initialize cv2.VideoCapture() object "
                                   f"with file {file}. The file may be corrupt or encoded into an unparseable file type.")
                sys.exit(1)
            
            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_color_shifted" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                                   f"Please ensure that {dir_file_path} is a valid path in your current working directory tree.")
                sys.exit(1)
            
            # Getting the video duration for weight calculations
            frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv.CAP_PROP_FPS)
            cap_duration = float(frame_count)/float(fps)

            if offset_t == 0.0:
                offset_t = cap_duration // 1
            
            timing_kwargs = dict({"end":offset_t}, **kwargs)

        # Main Processing loop for video files (will only iterate once over images)
        while True:
            frame = None
            if static_image_mode:
                frame = cv.imread(file)
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
            elif static_image_mode:
                logger.error("Face mesh detection error, function exiting with status 1.")
                debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process() on the current frame.")
                sys.exit(1)
            else:
                continue
            
            # Define an empty matlike in the shape of the frame, on which we will overlay our masks
            masked_frame = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)

            # Iterate over and mask all provided landmark regions
            for landmark_set in landmark_regions:

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
                        for cur_source, cur_target in LIPS_TIGHT_PATH:
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

                        # Creating boolean masks for the facial landmarks 
                        bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                        bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
                        bool_mask = bool_mask.astype(bool)

                        masked_frame[bool_mask] = 255
                        continue

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
            
            if not static_image_mode:
                # Getting the current video timestamp
                dt = capture.get(cv.CAP_PROP_POS_MSEC)/1000

                if dt < onset_t:
                    result.write(frame)
                elif dt < offset_t:
                    weight = timing_func(dt, **timing_kwargs)
                    frame_coloured = shift_color_temp(img=frame, img_mask=masked_frame, shift_weight=weight, shift_color=shift_color, max_color_shift=shift_magnitude)
                    frame_coloured[foreground == 0] = frame[foreground == 0]
                    result.write(frame_coloured)
                else:
                    dt = cap_duration - dt
                    weight = timing_func(dt, **timing_kwargs)
                    frame_coloured = shift_color_temp(img=frame, img_mask=masked_frame, shift_weight=weight, shift_color=shift_color, max_color_shift=shift_magnitude)
                    frame_coloured[foreground == 0] = frame[foreground == 0]
                    result.write(frame_coloured)
            
            else:
                frame_coloured = shift_color_temp(img=frame, img_mask=masked_frame, shift_weight=1.0, shift_color=shift_color, max_color_shift=shift_magnitude)
                frame_coloured[foreground == 0] = frame[foreground == 0]
                success = cv.imwrite(output_dir + "\\" + filename + "_color_shifted" + extension, frame_coloured)

                if not success:
                    logger.warning("Function has encountered an error attempting to call cv2.imwrite(), exiting with status 1.")
                    debug_logger.warning("Function has encountered an error attempting to call cv2.imwrite() to "
                                         f"output directory {dir_file_path}. Please ensure that this directory path is a valid path in your current working tree.")
                    sys.exit(1)
                break

        if not static_image_mode:
            capture.release()
            result.release()
        
        logger.info(f"Function execution completed successfully, view outputted files at {dir_file_path}.")

def face_saturation_shift(input_dir:str, output_dir:str, onset_t:float = 0.0, offset_t:float = 0.0, shift_magnitude:float = -8.0, 
                          timing_func:Callable[..., float] = sigmoid, landmark_regions:list[list[tuple]] | list[tuple] = FACE_SKIN_PATH, with_sub_dirs:bool = False, 
                          min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5, **kwargs) -> None:
    """For each image or video file contained in input_dir, the function applies a weighted saturation shift to the face region, 
    outputting each processed file to output_dir. Weights are calculated using a passed timing function, that returns
    a float in the normalised range [0,1].
    (NOTE there is currently no checking to ensure timing function outputs are normalised)

    Parameters
    ----------

    input_dir: str
        A path string to the directory containing input video files.

    output_dir: str
        A path string to the directory where outputted video files will be saved.
    
    onset_t: float
        The onset time of the colour shifting.
    
    offset_t: float
        The offset time of the colour shifting.
    
    shift_magnitude: float
        The maximum units to shift the saturation by, during peak onset.
    
    timingFunc: Function() -> float
        Any function that takes at least one input float (time), and returns a float.
    
    landmark_regions: list of list, list of tuple
        A list of one or more landmark paths, specifying the region in which the colouring will take place.
    
    with_sub_dirs: bool
        A boolean flag indicating whether the input directory contains nested directories.
    
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
    OSError
        Given invalid directory paths.
    ValueError:
        If provided timing_func does not return a normalised float value.
    """

    singleFile = False
    static_image_mode = False
    
    # Performing checks on function parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Face_saturation_shift: invalid type for parameter input_dir.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path in your current working tree.")
        raise OSError("Face_saturation_shift: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Face_saturation_shift: parameter output_dir must be a str.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path in your current working tree.")
        raise OSError("Face_saturation_shift: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered a ValueError for input parameter output_dir. "
                       "Message: output_dir is not a valid path to a directory in your current working tree.")
        raise ValueError("Face_saturation_shift: output_dir must be a valid path to a directory.")
    
    if not isinstance(onset_t, float):
        logger.warning("Function encountered a TypeError for input parameter onset_t. "
                       "Message: invalid type for parameter onset_t, expected float.")
        raise TypeError("Face_saturation_shift: parameter onset_t must be a float.")
    if not isinstance(offset_t, float):
        logger.warning("Function encountered a TypeError for input parameter offset_t. "
                       "Message: invalid type for parameter offset_t, expected float.")
        raise TypeError("Face_saturation_shift: parameter offset_t must be a float.")
    if not isinstance(shift_magnitude, float):
        logger.warning("Function encountered a TypeError for input parameter shift_magnitude. "
                       "Message: invalid type for parameter shift_magnitude, expected float.")
        raise TypeError("Face_saturation_shift: parameter shift_magnitude must be a float.")

    if not isinstance(landmark_regions, list):
        logger.warning("Function encountered a TypeError for input parameter landmark_regions. "
                       "Message: invalid type for parameter landmark_regions, expected list.")
        raise TypeError("Face_saturation_shift: parameter landmarks_to_color expects a list.")
    for val in landmark_regions:
        if not isinstance(val, list) or not isinstance(val, tuple):
            logger.warning("Function encountered a ValueError for input parameter landmark_regions. "
                           "Message: landmark_regions must either be a list[list[tuple]] or list[tuple].")
            raise ValueError("Face_saturation_shift: landmarks_to_color may either be a list of lists, or a singular list of tuples.")

    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Face_saturation_shift: parameter with_sub_dirs must be of type bool.")
    
    if not isinstance(min_detection_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_detection_confidence. "
                       "Message: invalid type for parameter min_detection_confidence, expected float.")
        raise TypeError("Face_saturation_shift: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_detection_confidence. "
                       "Message: min_detection_confidence must be a float in the range [0,1].")
        raise ValueError("Face_saturation_shift: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_tracking_confidence. "
                       "Message: invalid type for parameter min_tracking_confidence, expected float.")
        raise TypeError("Face_saturation_shift: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_tracking_confidence. "
                       "Message: min_tracking_confidence must be a float in the range [0,1].")
        raise ValueError("Face_saturation_shift: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Logging input parameters
    landmark_names = "["
    for i in range(len(landmark_regions)):
        cur_name = get_variable_name(landmark_regions[i], globals())
        if i == len(landmark_regions) - 1:
            landmark_names += f"{cur_name}]"
        else:
            landmark_names += f"{cur_name}, "

    logger.info(f"Input parameters: onset_t = {onset_t}, offset_t = {offset_t}, shift_magnitude = {shift_magnitude}, "
                f"landmark_regions = {landmark_names}, with_sub_dirs = {with_sub_dirs}.")
    logger.info(f"Mediapipe configurations: min_detection_confidence = {min_detection_confidence}, min_tracking_confidence = {min_tracking_confidence}.")
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = []

    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Sat_Shifted"):
        os.mkdir(output_dir + "\\Sat_Shifted")
        output_dir = output_dir + "\\Sat_Shifted"
        logger.info(f"Function created a new output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\Sat_Shifted"
    
    for file in files_to_process:
            
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        cap_duration = None
        dir_file_path = output_dir + f"{filename}_sat_shifted{extension}"

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
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
                sys.exit(1)
        
        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoCapture() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encounterd an error attempting to initialize cv2.VideoCapture() object "
                                   f"with file {file}. The file may be corrupt or encoded into an unparseable file type.")
                sys.exit(1)
            
            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_sat_shifted" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                                   f"Please ensure that {dir_file_path} is a valid path in your current working directory tree.")
                sys.exit(1)
            
            # Getting the video duration for weight calculations
            frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv.CAP_PROP_FPS)
            cap_duration = float(frame_count)/float(fps)

            if offset_t == 0.0:
                offset_t = cap_duration // 1

            timing_kwargs = dict({"end":offset_t}, **kwargs)

        while True:
            frame = None
            if static_image_mode:
                frame = cv.imread(file)
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
            elif static_image_mode:
                logger.error("Face mesh detection error, function exiting with status 1.")
                debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process() on the current frame.")
                sys.exit(1)
            else:
                continue
            
            masked_frame = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)

            # Iterate over and mask all provided landmark regions
            for landmark_set in landmark_regions:

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
                        for cur_source, cur_target in LIPS_TIGHT_PATH:
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

                        # Creating boolean masks for the facial landmarks 
                        bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                        bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
                        bool_mask = bool_mask.astype(bool)

                        masked_frame[bool_mask] = 255
                        continue
            
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
            
            if not static_image_mode:
                # Getting the current video timestamp
                dt = capture.get(cv.CAP_PROP_POS_MSEC)/1000

                if dt < onset_t:
                    result.write(frame)
                elif dt < offset_t:
                    shift_weight = timing_func(dt, **timing_kwargs)
                    img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)

                    h,s,v = cv.split(img_hsv)
                    s = np.where(masked_frame == 255, s + (shift_weight * shift_magnitude), s)
                    np.clip(s,0,255)
                    img_hsv = cv.merge([h,s,v])

                    img_bgr = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
                    img_bgr[foreground == 0] = frame[foreground == 0]
                    result.write(img_bgr)
                else:
                    dt = cap_duration - dt
                    shift_weight = timing_func(dt, **timing_kwargs)
                    img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)

                    h,s,v = cv.split(img_hsv)
                    s = np.where(masked_frame == 255, s + (shift_weight * shift_magnitude), s)
                    np.clip(s,0,255)
                    img_hsv = cv.merge([h,s,v])

                    img_bgr = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
                    img_bgr[foreground == 0] = frame[foreground == 0]
                    result.write(img_bgr)

            else:
                img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)

                h,s,v = cv.split(img_hsv)
                s = np.where(masked_frame == 255, s + (1.0 * shift_magnitude), s)
                np.clip(s,0,255)
                img_hsv = cv.merge([h,s,v])

                img_bgr = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
                img_bgr[foreground == 0] = frame[foreground == 0]
                success = cv.imwrite(output_dir + "\\" + filename + "_sat_shifted" + extension, img_bgr)

                if not success:
                    logger.warning("Function has encountered an error attempting to call cv2.imwrite(), exiting with status 1.")
                    debug_logger.warning("Function has encountered an error attempting to call cv2.imwrite() to "
                                         f"output directory {dir_file_path}. Please ensure that this directory path is a valid path in your current working tree.")
                    sys.exit(1)

                break

        if not static_image_mode:
            capture.release()
            result.release()

        logger.info(f"Function execution completed successfully, view outputted files at {dir_file_path}.")

def face_brightness_shift(input_dir:str, output_dir:str, onset_t:float = 0.0, offset_t:float = 0.0, shift_magnitude:int = 20, 
                        timing_func:Callable[..., float] = sigmoid, landmark_regions:list[list[tuple]] | list[tuple] = FACE_SKIN_PATH, with_sub_dirs:bool = False, 
                        min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5, **kwargs) -> None:
    """For each image or video file contained in input_dir, the function applies a weighted brightness shift to the face region, 
    outputting each processed file to output_dir. Weights are calculated using a passed timing function, that returns a float in the normalised range [0,1].
    (NOTE there is currently no checking to ensure timing function outputs are normalised)

    Parameters
    ----------

    input_dir: str
        A path string to the directory containing input video files.

    output_dir: str
        A path string to the directory where outputted video files will be saved.
    
    onset_t: float
        The onset time of the brightness shifting.
    
    offset_t: float
        The offset time of the brightness shifting.
    
    shift_magnitude: float
        The maximum units to shift the brightness by, during peak onset.
    
    timingFunc: Function() -> float
        Any function that takes at least one input float (time), and returns a float.
    
    landmark_regions: list of list, list of tuple
        A list of one or more landmark paths, specifying the region in which the colouring will take place.
    
    with_sub_dirs: bool
        A boolean flag indicating whether the input directory contains nested directories.
    
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
    OSError
        Given invalid directory paths.
    ValueError:
        If provided timing_func does not return a normalised float value.
    """

    singleFile = False
    static_image_mode = False

    # Performing checks on function parameters
    
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Face_brightness_shift: invalid type for parameter input_dir.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path in your current working tree.")
        raise OSError("Face_brightness_shift: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Face_brightness_shift: parameter output_dir must be a str.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path in your current working tree.")
        raise OSError("Face_brightness_shift: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered a ValueError for input parameter output_dir. "
                       "Message: output_dir is not a valid path to a directory in your current working tree.")
        raise ValueError("Face_brightness_shift: output_dir must be a valid path to a directory.")
    
    if not isinstance(onset_t, float):
        logger.warning("Function encountered a TypeError for input parameter onset_t. "
                       "Message: invalid type for parameter onset_t, expected float.")
        raise TypeError("Face_brightness_shift: parameter onset_t must be a float.")
    if not isinstance(offset_t, float):
        logger.warning("Function encountered a TypeError for input parameter offset_t. "
                       "Message: invalid type for parameter offset_t, expected float.")
        raise TypeError("Face_brightness_shift: parameter offset_t must be a float.")
    if not isinstance(shift_magnitude, float):
        logger.warning("Function encountered a TypeError for input parameter shift_magnitude. "
                       "Message: invalid type for parameter shift_magnitude, expected float.")
        raise TypeError("Face_brightness_shift: parameter shift_magnitude must be a float.")

    if not isinstance(landmark_regions, list):
        logger.warning("Function encountered a TypeError for input parameter landmark_regions. "
                       "Message: invalid type for parameter landmark_regions, expected list.")
        raise TypeError("Face_brightness_shift: parameter landmarks_to_color expects a list.")
    for val in landmark_regions:
        if not isinstance(val, list) or not isinstance(val, tuple):
            logger.warning("Function encountered a ValueError for input parameter landmark_regions. "
                           "Message: landmark_regions must either be a list[list[tuple]] or list[tuple].")
            raise ValueError("Face_brightness_shift: landmarks_to_color may either be a list of lists, or a singular list of tuples.")

    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Face_brightness_shift: parameter with_sub_dirs must be of type bool.")
    
    if not isinstance(min_detection_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_detection_confidence. "
                       "Message: invalid type for parameter min_detection_confidence, expected float.")
        raise TypeError("Face_brightness_shift: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_detection_confidence. "
                       "Message: min_detection_confidence must be a float in the range [0,1].")
        raise ValueError("Face_brightness_shift: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_tracking_confidence. "
                       "Message: invalid type for parameter min_tracking_confidence, expected float.")
        raise TypeError("Face_brightness_shift: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_tracking_confidence. "
                       "Message: min_tracking_confidence must be a float in the range [0,1].")
        raise ValueError("Face_brightness_shift: parameter min_tracking_confidence must be in the range [0,1].")
      
    # Creating a list of file path strings to iterate through when processing
    files_to_process = []

    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Brightness_Shifted"):
        os.mkdir(output_dir + "\\Brightness_Shifted")
    output_dir = output_dir + "\\Brightness_Shifted"
    
    for file in files_to_process:
            
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        cap_duration = None
        dir_file_path = output_dir + f"{filename}_bright_shifted{extension}"

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
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
                sys.exit(1)
        
        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoCapture() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encounterd an error attempting to initialize cv2.VideoCapture() object "
                                   f"with file {file}. The file may be corrupt or encoded into an unparseable file type.")
                sys.exit(1)
            
            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_bright_shifted" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                                   f"Please ensure that {dir_file_path} is a valid path in your current working directory tree.")
                sys.exit(1)
            
            # Getting the video duration for weight calculations
            frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv.CAP_PROP_FPS)
            cap_duration = float(frame_count)/float(fps)

            if offset_t == 0.0:
                offset_t = cap_duration // 1
            
            timing_kwargs = dict({"end":offset_t}, **kwargs)
            
        while True:
            frame = None
            if static_image_mode:
                frame = cv.imread(file)
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
            elif static_image_mode:
                logger.error("Face mesh detection error, function exiting with status 1.")
                debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process() on the current frame.")
                sys.exit(1)
            else:
                continue
            
            masked_frame = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)

            # Iterate over and mask all provided landmark regions
            for landmark_set in landmark_regions:

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
                        for cur_source, cur_target in LIPS_TIGHT_PATH:
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

                        # Creating boolean masks for the facial landmarks 
                        bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                        bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
                        bool_mask = bool_mask.astype(bool)

                        masked_frame[bool_mask] = 255
                        continue
            
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

            # Reshaping to allow numpy broadcasting
            masked_frame = masked_frame.reshape((masked_frame.shape[0], masked_frame.shape[1], 1))

            if not static_image_mode:
                # Getting the current video timestamp
                dt = capture.get(cv.CAP_PROP_POS_MSEC)/1000

                if dt < onset_t:
                    result.write(frame)
                elif dt < offset_t:
                    weight = timing_func(dt, **timing_kwargs)

                    img_brightened = frame.copy()
                    img_brightened = np.where(masked_frame == 255, cv.convertScaleAbs(src=img_brightened, alpha=1, beta=(weight * shift_magnitude)), frame)

                    img_brightened[foreground == 0] = frame[foreground == 0]
                    result.write(img_brightened)
                else:
                    dt = cap_duration - dt
                    weight = timing_func(dt, **timing_kwargs)

                    img_brightened = frame.copy()
                    img_brightened = np.where(masked_frame == 255, cv.convertScaleAbs(src=img_brightened, alpha=1, beta=(weight * shift_magnitude)), frame)

                    img_brightened[foreground == 0] = frame[foreground == 0]
                    result.write(img_brightened)

            else:
                # Brightening the image
                img_brightened = frame.copy()
                img_brightened = np.where(masked_frame == 255, cv.convertScaleAbs(src=img_brightened, alpha=1, beta=shift_magnitude), frame)

                # Making sure background remains unaffected
                img_brightened[foreground == 0] = frame[foreground == 0]

                success = cv.imwrite(output_dir + "\\" + filename + "_brightened" + extension, img_brightened)

                if not success:
                    logger.warning("Function has encountered an error attempting to call cv2.imwrite(), exiting with status 1.")
                    debug_logger.warning("Function has encountered an error attempting to call cv2.imwrite() to "
                                         f"output directory {dir_file_path}. Please ensure that this directory path is a valid path in your current working tree.")
                    sys.exit(1)

                break
        
        if not static_image_mode:
            capture.release()
            result.release()
        
        logger.info(f"Function execution completed successfully, view outputted files at {dir_file_path}.")