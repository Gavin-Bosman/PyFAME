from pyfame.util.util_constants import *
from pyfame.mesh import *
from pyfame.timing.apply_timing_function import *
from pyfame.util.util_general_utilities import get_variable_name
from pyfame.util.util_exceptions import *
from pyfame.io import *
from pyfame.manipulation.occlusion.apply_occlusion_overlay import get_mask_from_path, mask_frame
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
from typing import Callable
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def frame_saturation_shift(frame:cv.typing.MatLike, mask:cv.typing.MatLike | None, shift_weight:float, 
                           shift_magnitude:float = -8.0, static_image_mode:bool = False) -> cv.typing.MatLike:
    if mask is None:
        face_mesh = get_mesh(0.5, 0.5, static_image_mode, 1)
        mask = mask_frame(frame, face_mesh, FACE_OVAL_MASK, return_mask=True)

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

    img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)

    if static_image_mode:
        h,s,v = cv.split(img_hsv)
        s = np.where(mask == 255, s + (1.0 * shift_magnitude), s)
        np.clip(s,0,255)
        img_hsv = cv.merge([h,s,v])
    else:
        h,s,v = cv.split(img_hsv)
        s = np.where(mask == 255, s + (shift_weight * shift_magnitude), s)
        np.clip(s,0,255)
        img_hsv = cv.merge([h,s,v])

    img_bgr = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
    img_bgr[foreground == 0] = frame[foreground == 0]
    return img_bgr

def face_saturation_shift(input_dir:str, output_dir:str, onset_t:float = 0.0, offset_t:float = 0.0, shift_magnitude:float = -8.0, 
                          timing_func:Callable[..., float] = timing_linear, landmark_regions:list[list[tuple]] = [FACE_SKIN_PATH], with_sub_dirs:bool = False, 
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
        For the full list of available timing functions, please see pyfame_utils.display_timing_function_options().
    
    landmark_regions: list of list, list of tuple
        A list of one or more landmark paths, specifying the region in which the colouring will take place.
        For the full list of predefined landmark paths, please see pyfame_utils.display_all_landmark_paths().
    
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
    elif onset_t < 0:
        logger.warning("Function encountered a ValueError for input parameter onset_t. "
                       "Message: onset_t must be a positive float value.")
        raise ValueError("Face_saturation_shift: parameter onset_t must be a positive float.")
    
    if not isinstance(offset_t, float):
        logger.warning("Function encountered a TypeError for input parameter offset_t. "
                       "Message: invalid type for parameter offset_t, expected float.")
        raise TypeError("Face_saturation_shift: parameter offset_t must be a float.")
    elif offset_t < 0:
        logger.warning("Function encountered a ValueError for input parameter offset_t. "
                       "Message: offset_t must be a positive float value.")
        raise ValueError("Face_saturation_shift: parameter offset_t must be a positive float.")

    if not isinstance(shift_magnitude, float):
        logger.warning("Function encountered a TypeError for input parameter shift_magnitude. "
                       "Message: invalid type for parameter shift_magnitude, expected float.")
        raise TypeError("Face_saturation_shift: parameter shift_magnitude must be a float.")

    if not isinstance(landmark_regions, list):
        logger.warning("Function encountered a TypeError for input parameter landmark_regions. "
                       "Message: invalid type for parameter landmark_regions, expected list.")
        raise TypeError("Face_saturation_shift: parameter landmarks_to_color expects a list.")
    elif len(landmark_regions) == 0:
        logger.warning("Function encountered a ValueError for input parameter landmark_regions. "
                       "Message: landmark_regions cannot be an empty list.")
        raise ValueError("Face_saturation_shift: parameter landmark_regions cannot be an empty list.")
    for val in landmark_regions:
        if not isinstance(val, list) or not isinstance(val[0], tuple):
            logger.warning("Function encountered a ValueError for input parameter landmark_regions. "
                           "Message: landmark_regions must either be a list[list[tuple]].")
            raise TypeError("Face_saturation_shift: landmarks_regions expects a list of list of tuple.")

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

    # Declaring the mediapipe facemesh variable
    face_mesh = None
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
    
    logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")
    
    # Creating named output directories for video output
    output_dir = create_output_directory(output_dir, "Saturation_Shifted")
    
    for file in files_to_process:
            
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        cap_duration = None
        dir_file_path = output_dir + f"\\{filename}_sat_shifted{extension}"

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
                logger.error("Function has encountered an unparseable file type, Function exiting with status 1. " 
                             "Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                   "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                raise UnrecognizedExtensionError()
        
        if not static_image_mode:
            capture = get_video_capture(file) 
            size = (int(capture.get(3)), int(capture.get(4)))

            result = get_video_writer(dir_file_path, size, codec=codec)
            
            # Getting the video duration for weight calculations
            frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv.CAP_PROP_FPS)
            cap_duration = float(frame_count)/float(fps)

            if offset_t == 0.0:
                offset_t = cap_duration - 1.0

            timing_kwargs = dict({"end":offset_t}, **kwargs)

        while True:
            frame = None
            if static_image_mode:
                frame = cv.imread(file)
                if frame is None:
                    logger.error("Function encountered an error attempting to call cv2.imread(file).")
                    debug_logger.error("Function encountered an error attempting to call cv2.imread(file) "
                                       f"over file {file}.")
                    raise FileReadError()
            else:
                success, frame = capture.read()
                if not success:
                    break    

            masked_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            # Creating the frame mask
            for landmark_path in landmark_regions:
                mask = get_mask_from_path(frame, landmark_path, face_mesh)
                masked_frame = cv.bitwise_or(masked_frame, mask)
            
            if not static_image_mode:
                # Getting the current video timestamp
                dt = capture.get(cv.CAP_PROP_POS_MSEC)/1000

                if dt < onset_t:
                    result.write(frame)
                elif dt < offset_t:
                    shift_weight = timing_func(dt, **timing_kwargs)
                    img_out = frame_saturation_shift(frame, masked_frame, shift_weight, shift_magnitude, static_image_mode)
                    result.write(img_out)
                else:
                    dt = cap_duration - dt
                    shift_weight = timing_func(dt, **timing_kwargs)
                    img_out = frame_saturation_shift(frame, masked_frame, shift_weight, shift_magnitude, static_image_mode)
                    result.write(img_out)
            else:
                img_out = frame_saturation_shift(frame, masked_frame, shift_weight, shift_magnitude, static_image_mode)

                success = cv.imwrite(output_dir + "\\" + filename + "_sat_shifted" + extension, img_out)
                if not success:
                    logger.warning("Function has encountered an error attempting to call cv2.imwrite(), exiting with status 1.")
                    debug_logger.warning("Function has encountered an error attempting to call cv2.imwrite() to "
                                         f"output directory {dir_file_path}. Please ensure that this directory path is a valid path in your current working tree.")
                    raise FileWriteError()

                break

        if not static_image_mode:
            capture.release()
            result.release()

        logger.info(f"Function execution completed successfully, view outputted files at {dir_file_path}.")