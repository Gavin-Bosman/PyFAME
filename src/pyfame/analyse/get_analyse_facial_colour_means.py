from pyfame.mesh import get_mesh
from pyfame.mesh.mesh_landmarks import *
from pyfame.layer.manipulations.mask import mask_from_path
from pyfame.file_access import get_video_capture, get_directory_walk, create_output_directory
from pyfame.utilities.exceptions import *
from pyfame.utilities.constants import *
from pyfame.utilities.checks import *
import os
import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def analyse_facial_colour_means(input_directory:str, output_directory:str, color_space: int|str = COLOR_SPACE_BGR, with_sub_dirs:bool = False,
                           min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    """Takes an input video file, and extracts colour channel means in the specified color space for the full-face, cheeks, nose and chin.
    Creates a new directory 'Color_Channel_Means', where a csv file will be written to for each input video file provided.

    Parameters
    ----------

    input_directory: str
        A path string to a directory containing the video files to be processed.

    output_directory: str
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
        
    Returns
    -------

    None
    """
    
    # Global declarations and init
    static_image_mode = False

    # Type and value checking input parameters
    check_type(input_directory, [str])
    check_valid_path(input_directory)

    check_type(output_directory, [str])
    check_valid_path(output_directory)
    check_is_dir(output_directory)

    check_type(color_space, [int, str])
    check_value(color_space, ["bgr", "hsv", "greyscale", "grayscale"].extend(COLOR_SPACE_OPTIONS))
    if isinstance(color_space, str):
        str_map = {"bgr":COLOR_SPACE_BGR, "hsv":COLOR_SPACE_HSV, "greyscale":COLOUR_SPACE_GREYSCALE, "grayscale":COLOUR_SPACE_GREYSCALE}
        low_str = str.lower(color_space)
        color_space = str_map.get(low_str)

    check_type(with_sub_dirs, [bool])
    check_type(min_detection_confidence, [float])
    check_value(min_detection_confidence, min=0.0, max=1.0)

    check_type(min_tracking_confidence, [float])
    check_value(min_tracking_confidence, min=0.0, max=1.0)
    
    # Defining mediapipe facemesh task
    face_mesh = None
    
    # Creating a list of file path strings
    files_to_process = get_directory_walk(input_directory, with_sub_dirs)
    
    
    # Create an output directory for the csv files
    output_directory = create_output_directory(output_directory,"Color_Channel_Means")
    
    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        capture = None
        csv = None
        dir_file_path = output_directory

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
                raise UnrecognizedExtensionError()

        if not static_image_mode:
            capture = get_video_capture(file)
            
            # Writing the column headers to csv
            if color_space == COLOUR_SPACE_BGR:
                dir_file_path += f"\\{filename}_RGB.csv"
                csv = open(output_directory + "\\" + filename + "_RGB.csv", "w")
                csv.write("Timestamp,Mean_Red,Mean_Green,Mean_Blue,Cheeks_Red,Cheeks_Green,Cheeks_Blue," +
                        "Nose_Red,Nose_Green,Nose_Blue,Chin_Red,Chin_Green,Chin_Blue\n")
            elif color_space == COLOUR_SPACE_HSV:
                dir_file_path += f"\\{filename}_HSV.csv"
                csv = open(output_directory + "\\" + filename + "_HSV.csv", "w")
                csv.write("Timestamp,Mean_Hue,Mean_Sat,Mean_Value,Cheeks_Hue,Cheeks_Sat,Cheeks_Value," + 
                        "Nose_Hue,Nose_Sat,Nose_Value,Chin_Hue,Chin_Sat,Chin_Value\n")
            elif color_space == COLOUR_SPACE_GREYSCALE:
                dir_file_path += f"\\{filename}_GRAYSCALE.csv"
                csv = open(output_directory + "\\" + filename + "_GRAYSCALE.csv", "w")
                csv.write("Timestamp,Mean_Value,Cheeks_Value,Nose_Value,Chin_Value\n")
        else:
            # Writing the column headers to csv
            if color_space == COLOUR_SPACE_BGR:
                dir_file_path += f"\\{filename}_RGB.csv"
                csv = open(output_directory + "\\" + filename + "_RGB.csv", "w")
                csv.write("Mean_Red,Mean_Green,Mean_Blue,Cheeks_Red,Cheeks_Green,Cheeks_Blue," +
                        "Nose_Red,Nose_Green,Nose_Blue,Chin_Red,Chin_Green,Chin_Blue\n")
            elif color_space == COLOUR_SPACE_HSV:
                dir_file_path += f"\\{filename}_HSV.csv"
                csv = open(output_directory + "\\" + filename + "_HSV.csv", "w")
                csv.write("Mean_Hue,Mean_Sat,Mean_Value,Cheeks_Hue,Cheeks_Sat,Cheeks_Value," + 
                        "Nose_Hue,Nose_Sat,Nose_Value,Chin_Hue,Chin_Sat,Chin_Value\n")
            elif color_space == COLOUR_SPACE_GREYSCALE:
                dir_file_path += f"\\{filename}_GRAYSCALE.csv"
                csv = open(output_directory + "\\" + filename + "_GRAYSCALE.csv", "w")
                csv.write("Mean_Value,Cheeks_Value,Nose_Value,Chin_Value\n")
    
    while True:
        if not static_image_mode:
            success, frame = capture.read()
            if not success:
                break
        else:
            frame = cv.imread(file)
            if frame is None:
                raise FileReadError()
        
        # Creating landmark path variables
        lc_path = create_path(LEFT_CHEEK_IDX)
        rc_path = create_path(RIGHT_CHEEK_IDX)
        chin_path = create_path(CHIN_IDX)

        # Creating masks
        lc_mask = mask_from_path(frame, lc_path, face_mesh)
        rc_mask = mask_from_path(frame, rc_path, face_mesh)
        chin_mask = mask_from_path(frame, chin_path, face_mesh)
        fo_tight_mask = mask_from_path(frame, FACE_OVAL_TIGHT_PATH, face_mesh)
        le_mask = mask_from_path(frame, LEFT_EYE_PATH, face_mesh)
        re_mask = mask_from_path(frame, RIGHT_EYE_PATH, face_mesh)
        nose_mask = mask_from_path(frame, NOSE_PATH, face_mesh)
        mouth_mask = mask_from_path(frame, MOUTH_PATH, face_mesh)
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
            if color_space == COLOUR_SPACE_BGR:
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

            elif color_space == COLOUR_SPACE_HSV:
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
            
            elif color_space == COLOUR_SPACE_GREYSCALE:
                # Extracting the color channel means
                val, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_fo_mask)
                v_cheeks, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_cheeks_mask)
                v_nose, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_nose_mask)
                v_chin, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_chin_mask)

                # Get the current video timestamp
                timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000

                csv.write(f"{timestamp:.5f},{val:.5f},{v_cheeks:.5f},{v_nose:.5f},{v_chin:.5f}\n")
        else:
            if color_space == COLOUR_SPACE_BGR:
                # Extracting the color channel means
                blue, green, red, *_ = cv.mean(frame, bin_fo_mask)
                b_cheeks, g_cheeks, r_cheeks, *_ = cv.mean(frame, bin_cheeks_mask)
                b_nose, g_nose, r_nose, *_ = cv.mean(frame, bin_nose_mask)
                b_chin, g_chin, r_chin, *_ = cv.mean(frame, bin_chin_mask)

                csv.write(f"{red:.5f},{green:.5f},{blue:.5f}," +
                        f"{r_cheeks:.5f},{g_cheeks:.5f},{b_cheeks:.5f}," + 
                        f"{r_nose:.5f},{g_nose:.5f},{b_nose:.5f}," + 
                        f"{r_chin:.5f},{g_chin:.5f},{b_chin:.5f}\n")

            elif color_space == COLOUR_SPACE_HSV:
                # Extracting the color channel means
                hue, sat, val, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_fo_mask)
                h_cheeks, s_cheeks, v_cheeks, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_cheeks_mask)
                h_nose, s_nose, v_nose, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_nose_mask)
                h_chin, s_chin, v_chin, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_chin_mask)

                csv.write(f"{hue:.5f},{sat:.5f},{val:.5f}," +
                        f"{h_cheeks:.5f},{s_cheeks:.5f},{v_cheeks:.5f}," +
                        f"{h_nose:.5f},{s_nose:.5f},{v_nose:.5f}," + 
                        f"{h_chin:.5f},{s_chin:.5f},{v_chin:.5f}\n")
            
            elif color_space == COLOUR_SPACE_GREYSCALE:
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