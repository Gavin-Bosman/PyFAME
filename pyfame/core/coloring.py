from pyfame.utils.predefined_constants import *
from pyfame.utils.landmarks import *
from pyfame.utils.timing_functions import *
from pyfame.utils.utils import get_variable_name
from pyfame.core.exceptions import *
import os
import cv2 as cv
import cv2.typing
import mediapipe as mp
import numpy as np
from skimage.util import *
from typing import Callable
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def face_color_shift(input_dir:str, output_dir:str, onset_t:float = 0.0, offset_t:float = 0.0, shift_magnitude: float = 8.0, timing_func:Callable[...,float] = linear, 
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
        For the full list of available timing functions, please see pyfame_utils.display_timing_function_options().

    shift_color: str, int
        Either a string literal specifying the color of choice, or a predefined integer constant. For a full list of 
        available predefined values, please see pyfame_utils.display_shift_color_options().
    
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
            An integer or string literal specifying which color will be applied to the input image. For a full list of
            predifined options, please see pyfame_utils.display_shift_color_options().
                
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
                           "Message: unrecognized value for parameter shift_color, please see "
                           "pyfame_utils.display_shift_color_options() for a full list of accepted values.")
            raise ValueError("Face_color_shift: shift_color must be one of: red, green, blue, yellow.")
    elif isinstance(shift_color, int):
        if shift_color not in [COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW]:
            logger.warning("Function encountered a ValueError for input parameter shift_color. "
                           "Message: unrecognized value for parameter shift_color, please see "
                           "pyfame_utils.display_shift_color_options() for a full list of accepted values.")
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
        if not isinstance(val, list) and not isinstance(val, tuple):
            logger.warning("Function encountered a ValueError for input parameter landmark_regions. "
                           "Message: landmark_regions must either be a list[list[tuple]] or list[tuple].")
            raise ValueError("Face_color_shift: landmark_regions may either be a list of lists, or a singular list of tuples.")

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
        dir_file_path = output_dir + f"\\{filename}_color_shifted{extension}"

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
                raise UnrecognizedExtensionError()
        
        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoCapture() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encounterd an error attempting to initialize cv2.VideoCapture() object "
                                   f"with file {file}. The file may be corrupt or encoded into an unparseable file type.")
                raise FileReadError()
            
            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_color_shifted" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                                   f"Please ensure that {dir_file_path} is a valid path in your current working directory tree.")
                raise FileWriteError()
            
            # Getting the video duration for weight calculations
            frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv.CAP_PROP_FPS)
            cap_duration = float(frame_count)/float(fps)

            if offset_t == 0.0:
                offset_t = cap_duration - 1.0
            
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
                raise FaceNotFoundError()
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
                    raise FileWriteError()
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
        if not isinstance(val, list) and not isinstance(val, tuple):
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
        dir_file_path = output_dir + f"\\{filename}_sat_shifted{extension}"

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
                raise UnrecognizedExtensionError()
        
        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoCapture() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encounterd an error attempting to initialize cv2.VideoCapture() object "
                                   f"with file {file}. The file may be corrupt or encoded into an unparseable file type.")
                raise FileReadError()
            
            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_sat_shifted" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                                   f"Please ensure that {dir_file_path} is a valid path in your current working directory tree.")
                raise FileWriteError()
            
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
                raise FaceNotFoundError()
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
                    raise FileWriteError()

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
        if not isinstance(val, list) and not isinstance(val, tuple):
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
        dir_file_path = output_dir + f"\\{filename}_bright_shifted{extension}"

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
                raise UnrecognizedExtensionError()
        
        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoCapture() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encounterd an error attempting to initialize cv2.VideoCapture() object "
                                   f"with file {file}. The file may be corrupt or encoded into an unparseable file type.")
                raise FileReadError()
            
            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_bright_shifted" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                                   f"Please ensure that {dir_file_path} is a valid path in your current working directory tree.")
                raise FileWriteError()
            
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
                raise FaceNotFoundError()
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
                    raise FileWriteError()

                break
        
        if not static_image_mode:
            capture.release()
            result.release()
        
        logger.info(f"Function execution completed successfully, view outputted files at {dir_file_path}.")