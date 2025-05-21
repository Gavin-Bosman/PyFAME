from pyfame.util.util_constants import *
from pyfame.mesh import *
from pyfame.io import *
from pyfame.manipulation.occlusion.apply_occlusion_overlay import get_mask_from_path
from pyfame.util.util_general_utilities import get_variable_name, create_path
from pyfame.util.util_exceptions import *
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def get_facial_color_means(input_dir:str, output_dir:str, color_space: int|str = COLOR_SPACE_BGR, with_sub_dirs:bool = False,
                             min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    """Takes an input video file, and extracts colour channel means in the specified color space for the full-face, cheeks, nose and chin.
    Creates a new directory 'Color_Channel_Means', where a csv file will be written to for each input video file provided.

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
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Extract_color_channel_means: output_dir must be a path string.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path in your current working tree.")
        raise OSError("Extract_color_channel_means: output_dir is not a valid path.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered an ValueError for input parameter output_dir. "
                       "Message: output_dir is not a valid path to a directory in your current working tree.")
        raise ValueError("Extract_color_channel_means: output_dir must be a path string to a directory.")
    
    if isinstance(color_space, int):
        if color_space not in COLOR_SPACE_OPTIONS:
            logger.warning("Function encountered a ValueError for input parameter color_space. "
                           "Message: unrecognized value for parameter color_space, see pyfame_utils.display_color_space_options() "
                           "for a full list of accepted values.")
            raise ValueError("Extract_color_channel_means: unrecognized value for parameter color_space.")
    elif isinstance(color_space, str):
        if str.lower(color_space) not in ["rgb", "hsv", "grayscale"]:
            logger.warning("Function encountered a ValueError for input parameter color_space. "
                           "Message: unrecognized value for parameter color_space, see pyfame_utils.display_color_space_options() "
                           "for a full list of accepted values.")
            raise ValueError("Extract_color_channel_means: unrecognized value for parameter color_space.")
        else:
            if str.lower(color_space) == "rgb":
                color_space = COLOR_SPACE_BGR
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
    face_mesh = None
    
    # Creating a list of file path strings
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
    
    logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")
    
    # Create an output directory for the csv files
    output_dir = create_output_directory(output_dir,"Color_Channel_Means")
    
    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        capture = None
        csv = None
        dir_file_path = output_dir

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, static_image_mode)
            case ".mov":
                face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, static_image_mode)
            case ".png" | ".jpg" | ".jpeg" | ".bmp":
                static_image_mode = True
                face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, static_image_mode)
            case _:
                logger.error("Function has encountered an unparseable file type, Function exiting with status 1. " 
                             "Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                   "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                raise UnrecognizedExtensionError()

        if not static_image_mode:
            capture = get_video_capture(file)
            
            # Writing the column headers to csv
            if color_space == COLOR_SPACE_BGR:
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
            if color_space == COLOR_SPACE_BGR:
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
                raise FileReadError()
        
        # Creating landmark path variables
        lc_path = create_path(LEFT_CHEEK_IDX)
        rc_path = create_path(RIGHT_CHEEK_IDX)
        chin_path = create_path(CHIN_IDX)

        # Creating masks
        lc_mask = get_mask_from_path(frame, lc_path, face_mesh)
        rc_mask = get_mask_from_path(frame, rc_path, face_mesh)
        chin_mask = get_mask_from_path(frame, chin_path, face_mesh)
        fo_tight_mask = get_mask_from_path(frame, FACE_OVAL_TIGHT_PATH, face_mesh)
        le_mask = get_mask_from_path(frame, LEFT_EYE_PATH, face_mesh)
        re_mask = get_mask_from_path(frame, RIGHT_EYE_PATH, face_mesh)
        nose_mask = get_mask_from_path(frame, NOSE_PATH, face_mesh)
        mouth_mask = get_mask_from_path(frame, MOUTH_PATH, face_mesh)
        masks = [lc_mask, rc_mask, chin_mask, fo_tight_mask, le_mask, re_mask, nose_mask, mouth_mask]
        
        # Convert masks to binary representation
        for mask in masks:
            mask = mask.astype(bool)

        # Create binary image masks 
        bin_fo_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        bin_fo_mask[fo_tight_mask] = 255
        bin_fo_mask[le_mask] = 0
        bin_fo_mask[le_mask] = 0
        bin_fo_mask[mouth_mask] = 0

        bin_cheeks_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        bin_cheeks_mask[lc_mask] = 255
        bin_cheeks_mask[rc_mask] = 255

        bin_nose_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        bin_nose_mask[nose_mask] = 255

        bin_chin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        bin_chin_mask[chin_mask] = 255

        if not static_image_mode: 
            if color_space == COLOR_SPACE_BGR:
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
            if color_space == COLOR_SPACE_BGR:
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