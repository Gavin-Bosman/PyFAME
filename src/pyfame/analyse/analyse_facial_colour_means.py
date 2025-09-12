from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, PositiveFloat, PositiveInt
from typing import Union
from pyfame.mesh import get_mesh
from pyfame.mesh.mesh_landmarks import *
from pyfame.layer.manipulations.mask import mask_from_path
from pyfame.file_access import get_video_capture, create_output_directory
from pyfame.utilities.exceptions import *
from pyfame.utilities.constants import *
import os
import cv2 as cv
import numpy as np
import pandas as pd

class ColourMeansParameters(BaseModel):
    colour_space:Union[str, int]
    min_detection_confidence:PositiveFloat
    min_tracking_confidence:PositiveFloat
    output_sample_frequency_msec:PositiveInt

    @field_validator("colour_space", mode="before")
    @classmethod
    def check_recognized_value(cls, value, info:ValidationInfo):
        field_name = info.field_name
        value_mapping = {
            "bgr":COLOR_SPACE_BGR, 
            "rgb":COLOR_SPACE_BGR, 
            "hsv":COLOR_SPACE_HSV, 
            "greyscale":COLOUR_SPACE_GREYSCALE,
            "grayscale":COLOUR_SPACE_GREYSCALE
        }

        if isinstance(value, str):
            value = str.lower(value)
            if value not in {"rgb", "bgr", "hsv", "greyscale", "grayscale"}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value_mapping.get(value)
        
        if isinstance(value, int):
            if value not in {47, 48, 49}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        raise TypeError(f"Parameter {field_name} expects int or str.")
    
    @field_validator("min_detection_confidence", "min_tracking_confidence")
    @classmethod
    def check_valid_range(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if not (0.0 < value <= 1.0):
            raise ValueError(f"Invalid value for parameter {field_name}. The value must lie in the range [0, 1].")
        
        return value

def analyse_facial_colour_means(file_paths:pd.DataFrame, colour_space:int|str = COLOUR_SPACE_BGR, min_detection_confidence:float = 0.5, 
                                min_tracking_confidence:float = 0.5, output_sample_frequency_msec:int = 1000) -> dict[str, pd.DataFrame]:
    """Takes an input video file, and extracts colour channel means in the specified color space for the full-face, cheeks, nose and chin.
    Creates a new directory 'Color_Channel_Means', where a csv file will be written to for each input video file provided.

    Parameters
    ----------

    file_paths: Dataframe
        A 2-column dataframe consisting of absolute and relative file paths.
    
    colour_space: int, str
        A specifier for which color space to operate in. One of COLOR_SPACE_RGB, COLOR_SPACE_HSV or COLOR_SPACE_GRAYSCALE
    
    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.

    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
    
    output_sample_frequency_msec: int
        The time delay in milliseconds between successive csv write calls. Increase this value to speed up computation time, and decrease 
        the value to increase the number of optical flow vector samples written to the output csv file.
    
        
    Returns
    -------

    dict[str, pandas.Dataframe]

    Raises
    ------

    ValidationError:
        Thrown by the pydantic model when invalid parameters are passed to the method.
    
    FileReadError:
        When the working directory path; or any of its required sub-paths cannot be located. 
    
    UnrecognizedExtensionError
        If an image or video file is passed that is encoded in an unrecognized codec.
    """
    
    # Global declarations and init
    static_image_mode = False

    # Validate input parameters
    try:
        input_params = ColourMeansParameters(
            colour_space=colour_space,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {analyse_facial_colour_means.__name__}: {e}")
    
    colour_space = input_params.colour_space
    min_detection_confidence = input_params.min_detection_confidence
    min_tracking_confidence = input_params.min_tracking_confidence
    
    # Defining mediapipe facemesh task
    face_mesh = None

    # Extracting the i/o paths from the file_paths dataframe
    absolute_paths = file_paths["Absolute Path"]

    norm_path = os.path.normpath(absolute_paths.iloc[0])
    norm_cwd = os.path.normpath(os.getcwd())
    rel_dir_path, *_ = os.path.split(os.path.relpath(norm_path, norm_cwd))
    parts = rel_dir_path.split(os.sep)
    root_directory = None

    if parts is not None:
        root_directory = parts[0]
    
    if root_directory is None:
        root_directory = "data"
    
    test_path = os.path.join(norm_cwd, root_directory)

    if not os.path.isdir(test_path):
        raise FileReadError(message=f"Unable to locate the input {root_directory} directory. Please call make_output_paths() to set up the correct directory structure.")
    if not os.path.isdir(os.path.join(test_path, "raw")):
        raise FileReadError(message=f"Unable to locate the 'raw' subdirectory under root directory '{root_directory}'. Please call make_output_paths() to set up the correct directory structure.")
    if not os.path.isdir(os.path.join(test_path, "processed")):
        raise FileReadError(message=f"Unable to locate the 'processed' subdirectory under root directory '{root_directory}'. Please call make_output_paths() to set up the correct directory structure.")
    
    # Create an output directory for the csv files
    output_directory = os.path.join(test_path, "processed")
    output_directory = create_output_directory(output_directory,"colour_channel_means")

    outputs = {}
    
    for file in absolute_paths:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        capture = None

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4" | ".mov":
                face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, static_image_mode)
            case ".png" | ".jpg" | ".jpeg" | ".bmp":
                static_image_mode = True
                face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, static_image_mode)
            case _:
                raise UnrecognizedExtensionError()

        if not static_image_mode:
            capture = get_video_capture(file)
        
        timestamps = []

        # RGB value lists
        if colour_space == COLOUR_SPACE_BGR:
            red_means = []
            green_means = []
            blue_means = []
            cheeks_red_means = []
            cheeks_green_means = []
            cheeks_blue_means = []
            nose_red_means = []
            nose_green_means = []
            nose_blue_means = []
            chin_red_means = []
            chin_green_means = []
            chin_blue_means = []
        
        # HSV value lists
        elif colour_space == COLOUR_SPACE_HSV:
            hue_means = []
            sat_means = []
            val_means = []
            cheeks_hue_means = []
            cheeks_sat_means = []
            cheeks_val_means = []
            nose_hue_means = []
            nose_sat_means = []
            nose_val_means = []
            chin_hue_means = []
            chin_sat_means = []
            chin_val_means = []
        
        # Greyscale value lists
        else:
            grey_means = []
            cheeks_grey_means = []
            nose_grey_means = []
            chin_grey_means = []
        
        rolling_time_window = output_sample_frequency_msec
    
        while True:
            if not static_image_mode:
                success, frame = capture.read()
                if not success:
                    break
            else:
                frame = cv.imread(file)
                if frame is None:
                    raise FileReadError()
                
            timestamp = capture.get(cv.CAP_PROP_POS_MSEC)

            if timestamp < rolling_time_window:
                continue
            
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
             
            if colour_space == COLOUR_SPACE_BGR:
                # Extracting the color channel means
                blue, green, red, *_ = cv.mean(frame, bin_fo_mask)
                b_cheeks, g_cheeks, r_cheeks, *_ = cv.mean(frame, bin_cheeks_mask)
                b_nose, g_nose, r_nose, *_ = cv.mean(frame, bin_nose_mask)
                b_chin, g_chin, r_chin, *_ = cv.mean(frame, bin_chin_mask)

                timestamps.append(timestamp/1000)
                red_means.append(red)
                green_means.append(green)
                blue_means.append(blue)
                cheeks_red_means.append(r_cheeks)
                cheeks_green_means.append(g_cheeks)
                cheeks_blue_means.append(b_cheeks)
                nose_red_means.append(r_nose)
                nose_green_means.append(g_nose)
                nose_blue_means.append(b_nose)
                chin_red_means.append(r_chin)
                chin_green_means.append(g_chin)
                chin_blue_means.append(b_chin)

            elif colour_space == COLOUR_SPACE_HSV:
                # Extracting the color channel means
                hue, sat, val, *_ = cv.mean(cv.cvtColor(frame, cv.COLOR_BGR2HSV), bin_fo_mask)
                h_cheeks, s_cheeks, v_cheeks, *_ = cv.mean(cv.cvtColor(frame, cv.COLOR_BGR2HSV), bin_cheeks_mask)
                h_nose, s_nose, v_nose, *_ = cv.mean(cv.cvtColor(frame, cv.COLOR_BGR2HSV), bin_nose_mask)
                h_chin, s_chin, v_chin, *_ = cv.mean(cv.cvtColor(frame, cv.COLOR_BGR2HSV), bin_chin_mask)

                timestamps.append(timestamp/1000)
                hue_means.append(hue)
                sat_means.append(sat)
                val_means.append(val)
                cheeks_hue_means.append(h_cheeks)
                cheeks_sat_means.append(s_cheeks)
                cheeks_val_means.append(v_cheeks)
                nose_hue_means.append(h_nose)
                nose_sat_means.append(s_nose)
                nose_val_means.append(v_nose)
                chin_hue_means.append(h_chin)
                chin_sat_means.append(s_chin)
                chin_val_means.append(v_chin)
            
            else:
                # Extracting the color channel means
                val, *_ = cv.mean(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), bin_fo_mask)
                v_cheeks, *_ = cv.mean(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), bin_cheeks_mask)
                v_nose, *_ = cv.mean(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), bin_nose_mask)
                v_chin, *_ = cv.mean(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), bin_chin_mask)

                timestamps.append(timestamp/1000)
                grey_means.append(val)
                cheeks_grey_means.append(v_cheeks)
                nose_grey_means.append(v_nose)
                chin_grey_means.append(v_chin)
            
            rolling_time_window += output_sample_frequency_msec
        
        if not static_image_mode:
            capture.release()
        
        if colour_space == COLOUR_SPACE_BGR:
            output_df = pd.DataFrame({
                "timestamp":timestamps,
                "mean red":red_means,
                "mean green":green_means,
                "mean blue":blue_means,
                "mean cheeks red": cheeks_red_means,
                "mean cheeks green": cheeks_green_means,
                "mean cheeks blue": cheeks_blue_means,
                "mean nose red": nose_red_means,
                "mean nose green": nose_green_means,
                "mean nose blue": nose_blue_means,
                "mean chin red": chin_red_means,
                "mean chin green": chin_green_means,
                "mean chin blue": chin_blue_means
            })

            outputs.update({f"{filename}":output_df})

        elif colour_space == COLOUR_SPACE_HSV:
            output_df = pd.DataFrame({
                "timestamp":timestamps,
                "mean hue": hue_means,
                "mean saturation": sat_means,
                "mean value": val_means,
                "mean cheeks hue": cheeks_hue_means,
                "mean cheeks saturation": cheeks_sat_means,
                "mean cheeks value": cheeks_val_means,
                "mean nose hue": nose_hue_means,
                "mean nose saturation": nose_sat_means,
                "mean nose value": nose_val_means,
                "mean chin hue": chin_hue_means,
                "mean chin saturation": chin_sat_means,
                "mean chin value": chin_val_means
            })

            outputs.update({f"{filename}":output_df})

        else:
            output_df = pd.DataFrame({
                "timestamp":timestamps,
                "mean value": grey_means,
                "mean cheeks value": cheeks_grey_means,
                "mean nose value": nose_grey_means,
                "mean chin value": chin_grey_means
            })

            outputs.update({f"{filename}":output_df})

    return outputs    