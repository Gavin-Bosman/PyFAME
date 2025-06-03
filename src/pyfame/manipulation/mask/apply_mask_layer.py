from pyfame.util.util_constants import *
from pyfame.util.util_exceptions import *
from pyfame.util.util_general_utilities import get_variable_name
from pyfame.io import *
from pyfame.mesh import *
import os
import numpy as np
import cv2 as cv
import mediapipe as mp
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def layer_mask(frame: cv.typing.MatLike, face_mesh:mp.solutions.face_mesh.FaceMesh, roi:list[list[tuple]], 
               **kwargs) -> cv.typing.MatLike:
        
        # initializing background_color param
        background_color = (0,0,0)
        if kwargs.get("background_color") is not None:
            background_color = kwargs.get("background_color")
        background_color = np.asarray(background_color, dtype=np.uint8)

        # Generating the mask
        mask = get_mask_from_path(frame, roi, face_mesh)
        
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
        masked_frame = cv.bitwise_and(mask, foreground)
        masked_frame = np.reshape(masked_frame, (masked_frame.shape[0], masked_frame.shape[1], 1))
        masked_frame = np.where(masked_frame == 255, frame, background_color)
        return masked_frame

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
    face_mesh = None

    # Creating a list of file names to iterate through when processing
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
        
    logger.info(f"Function has read {len(files_to_process)} file(s) from input directory {input_dir}.")
    
    # Creating named output directories for video output
    output_dir = create_output_directory(output_dir, "Masked")

    for file in files_to_process:

        # Sniffing input filetype to determine running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        dir_file_path = output_dir + "\\" + filename + "_masked" + extension

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "mp4v"
                face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, static_image_mode)
            case ".mov":
                codec = "mp4v"
                face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, static_image_mode)
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                static_image_mode = True
                face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, static_image_mode)
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
                
                masked_frame = layer_mask(frame, face_mesh, mask_type, background_color)
                result.write(masked_frame)
        
            capture.release()
            result.release()
            logger.info(f"Function execution completed successfully. View outputted file(s) at {dir_file_path}.")
        
        else:
            img = cv.imread(file)
            if img is None:
                raise FileReadError()
            
            masked_img = layer_mask(img, face_mesh, mask_type, background_color)
            success = cv.imwrite(dir_file_path, masked_img)

            if not success:
                logger.error("Function encountered an FileWriteError attempting to call cv2.imwrite(). ")
                debug_logger.error("Function encountered an FileWriteError while attempting to call cv2.imwrite(). " 
                                   f"Ensure output_dir path string is valid, and ensure {file} is not corrupt.")
                raise FileWriteError()
            else:
                logger.info(f"Function execution completed successfully. View outputted file(s) at {output_dir}.")