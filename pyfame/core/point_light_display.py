from pyfame.utils.predefined_constants import *
from pyfame.utils.landmarks import *
from pyfame.utils.utils import get_variable_name
from pyfame.core.exceptions import *
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
from skimage.util import *
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def generate_point_light_display(input_dir:str, output_dir:str, landmark_regions:list[list[tuple]] = [FACE_OVAL_PATH], point_density:float = 0.5, 
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
                codec = "mp4v"
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "mp4v"
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case _:
                logger.error("Function has encountered an unparseable file type, Function exiting with status 1. " 
                             "Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                    "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                raise UnrecognizedExtensionError()
        
        capture = cv.VideoCapture(file)
        if not capture.isOpened():
            logger.error("Function has encountered an error attempting to initialize cv2.VideoCapture() object. "
                        "Function exiting with status 1.")
            debug_logger.error("Function has encounterd an error attempting to initialize cv2.VideoCapture() object "
                            f"with file {file}. The file may be corrupt or encoded into an unparseable file type.")
            raise FileReadError()
        
        size = (int(capture.get(3)), int(capture.get(4)))

        result = cv.VideoWriter(output_dir + "\\" + filename + "_pld" + extension,
                                cv.VideoWriter.fourcc(*codec), 30, size)
        if not result.isOpened():
            logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                        "Function exiting with status 1.")
            debug_logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                            f"Please ensure that {dir_file_path} is a valid path in your current working directory tree.")
            raise FileWriteError()
        
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
                logger.error("Function encountered a FaceMesh detection error attempting to call FaceMesh.process().")
                debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process() on the current frame.")
                raise FaceNotFoundError()
            
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
                            for cur_source, cur_target in MOUTH_PATH:
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
