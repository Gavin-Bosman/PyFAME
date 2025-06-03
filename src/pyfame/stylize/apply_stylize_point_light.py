from pyfame.util.util_constants import *
from pyfame.mesh import get_mesh_coordinates, get_mask_from_path
from pyfame.mesh.get_mesh_landmarks import *
from pyfame.util.util_general_utilities import get_variable_name
from pyfame.util.util_exceptions import *
from pyfame.io import get_video_capture, get_video_writer, get_directory_walk, create_output_directory
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
from skimage.util import *
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def generate_point_light_display(input_dir:str, output_dir:str, landmark_regions:list[list[tuple]] = [FACE_OVAL_PATH], point_density:float = 1.0, 
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

    # Perform checks on input parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Point_light_display: parameter input_dir expects a string.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path string to a directory or file.")
        raise OSError("Point_light_display: parameter input_dir is required to be a valid pathstring.")
    
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
        raise ValueError("Point_light_display: parameter output_dir must be a path string to a directory.")
    
    if not isinstance(landmark_regions, list):
        logger.warning("Function encountered a TypeError for input parameter landmark_regions. "
                       "Message: invalid type for parameter landmark_regions, expected list.")
        raise TypeError("Point_light_display: parameter landmark_regions expects a list of list.")
    elif len(landmark_regions) == 0:
        logger.warning("Function encountered a ValueError for input parameter landmark_regions. "
                       "Message: landmark_regions cannot be an empty list.")
        raise ValueError("Point_light_display: parameter landmark_regions cannot be an empty list.")
    for val in landmark_regions:
        if not isinstance(val, list) or not isinstance(val[0], tuple):
            logger.warning("Function encountered a TypeError for input parameter landmark_regions. "
                    "Message: invalid type for parameter landmark_regions, expected list of list of tuple.")
            raise TypeError("Point_light_display: parameter landmark_regions expects a list of list of tuple.")
    
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
                       "Message: unrecognized value for parameter history_mode, please see utils.display_history_mode_options() "
                       "for a full list of accepted values.")
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
    face_mesh = None
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
        
    logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")
    
    # Creating named output directories for video output
    output_dir = create_output_directory(output_dir, "PLD")

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
                codec = "mp4v"
                face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, False)
            case ".mov":
                codec = "mp4v"
                face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, False)
            case _:
                logger.error("Function has encountered an unparseable file type, Function exiting with status 1. " 
                             "Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                    "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                raise UnrecognizedExtensionError()
        
        capture = get_video_capture(file)
        size = (int(capture.get(3)), int(capture.get(4)))
        result = get_video_writer(dir_file_path, size, codec=codec)
        
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
            raise FileReadError()

        mask = np.zeros_like(frame, dtype=np.uint8)
        fps = capture.get(cv.CAP_PROP_FPS)
        frame_history_count = round(fps * (history_window_msec/1000))
        frame_history = []

        # Main Processing loop for video files
        while True:

            output_img = np.zeros_like(frame, dtype=np.uint8)

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            landmark_screen_coords = get_mesh_coordinates(frame_rgb, face_mesh)
            
            if counter == 0:
                for lm_path in landmark_regions:
                    lm_mask = get_mask_from_path(frame, lm_path, face_mesh)
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