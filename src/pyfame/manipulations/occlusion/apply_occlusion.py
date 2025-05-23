from pyfame.utils.predefined_constants import *
from pyfame.mesh.landmarks import *
from pyfame.utils.utils import get_variable_name, compute_rot_angle
from pyfame.utils.exceptions import *
from pyfame.io import *
from .apply_mask import get_mask_from_path
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
from skimage.util import *
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def occlude_frame(frame:cv.typing.MatLike, mask:cv.typing.MatLike, occlusion_fill:int, landmark_pixel_coords:list[dict]) -> cv.typing.MatLike:
        
        masked_frame = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

        match occlusion_fill:
            case 8 | 10:
                frame = np.where(masked_frame == 255, 0, frame)
                return frame

            case 9:
                cur_landmark_coords = []
                for cur_source, cur_target in FACE_OVAL_TIGHT_PATH:
                    source = landmark_pixel_coords[cur_source]
                    target = landmark_pixel_coords[cur_target]
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
                mean_img = np.zeros_like(frame, dtype=np.uint8)
                mean_img[:] = mean[:3]
                frame = np.where(masked_frame == 255, mean_img, frame)
                return frame

def occlude_face_region(input_dir:str, output_dir:str, landmarks_to_occlude:list[list[tuple]] = [BOTH_EYES_PATH], occlusion_fill:int = OCCLUSION_FILL_BLACK,
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
        predefined set of facial landmark paths. To see the full list of predefined landmark paths, please see 
        pyfame_utils.display_all_landmark_paths().
    
    occlusion_fill: int
        An integer flag indicating the fill method of the occluded landmark regions. One of OCCLUSION_FILL_BLACK or 
        OCCLUSION_FILL_MEAN. For a full list of available options please see pyfame_utils.display_occlusion_fill_options().
    
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
    FileReadError
        When an error is encountered instantiating VideoCapture or calling imRead.
    FileWriteError
        When an error is encountered instantiating VideoWriter or calling imWrite.
    UnrecognizedExtensionError
        When an unrecognized image or video file extension is provided.
    '''
    
    static_image_mode = False

    # Performing checks on function parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Occlude_face_region: parameter input_dir must be a str.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path, or the specified directory does not exist.")
        raise OSError("Occlude_face_region: input directory path is not a valid path, or the directory does not exist.")
    
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
            if not isinstance(val[0], tuple):
                logger.warning("Function encountered a ValueError for input parameter landmarks_to_occlude. "
                               "Message: parameter landmarks_to_occlude expects list[list[tuple]].")
                raise ValueError("Occlude_face_region: landmarks_to_occlude must be a list of list of tuples.")
    
    if not isinstance(occlusion_fill, int):
        logger.warning("Function encountered a TypeError for input parameter occlusion_fill. "
                       "Message: invalid type for parameter occlusion_fill, expected int.")
        raise TypeError("Occlude_face_region: parameter occlusion_fill must be of type int.")
    elif occlusion_fill not in [OCCLUSION_FILL_BLACK, OCCLUSION_FILL_MEAN, OCCLUSION_FILL_BAR]:
        logger.warning("Function encountered a ValueError for input parameter occlusion_fill. "
                       f"Message: {occlusion_fill} is not a valid option for parameter occlusion_fill. "
                       "Please see pyfame_utils.display_occlusion_fill_options().")
        raise ValueError("Occlude_face_region: parameter occlusion_fill must be one of OCCLUSION_FILL_BLACK, OCCLUSION_FILL_MEAN or OCCLUSION_FILL_BAR.")
    if occlusion_fill == OCCLUSION_FILL_BAR:
        print("\nWARNING: OCCLUSION_FILL_BAR is only compatible with BOTH_EYES_PATH, MOUTH_PATH and NOSE_PATH. While the function will occlude"
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
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
    
    logger.info(f"Function has read {len(files_to_process)} file(s) from input directory {input_dir}.")
    
    output_dir = create_output_dir(output_dir, "Occluded")

    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        dir_file_path = output_dir + f"\\{filename}_occluded{extension}"
        codec = None

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "mp4v"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "mp4v"
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
                raise UnrecognizedExtensionError()

        capture = None
        result = None
        min_x_lm = -1
        max_x_lm = -1
        prev_slope = -1

        if not static_image_mode:
            capture = get_video_capture(file)
            size = (int(capture.get(3)), int(capture.get(4)))

            result = get_video_writer(dir_file_path, size, codec=codec)

        while True:

            if static_image_mode:
                frame = cv.imread(file)
                if frame is None:
                    raise FileReadError()
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
                debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process().")
                logger.error("Face mesh detection error, function exiting with status 1.")
                raise FaceNotFoundError()
            
            masked_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            # Creating the frame mask
            for landmark_path in landmarks_to_occlude:
                if occlusion_fill == OCCLUSION_FILL_BAR:
                    cur_landmark_coords = []
                    # Converting landmark coords to screen coords
                    for cur_source, cur_target in landmark_path:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        cur_landmark_coords.append((source.get('x'),source.get('y')))
                        cur_landmark_coords.append((target.get('x'),target.get('y')))
                        
                    if min_x_lm < 0 or max_x_lm < 0:
                        min_x = 1000
                        max_x = 0

                        # find the two points closest to the beginning and end x-positions of the landmark region
                        unique_landmarks = np.unique(landmark_path)
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
                        rot_angle = compute_rot_angle(slope1=slope)
                        
                        rectangle = cv.rectangle(masked_frame, (p1.get('x')-50, cx - 50), (p2.get('x') + 50, cx + 50), (255,255,255), -1)
                        masked_frame_t = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

                        rot_mat = cv.getRotationMatrix2D((cy,cx), rot_angle, 1)
                        rot_img = cv.warpAffine(rectangle, rot_mat, (masked_frame_t.shape[1], masked_frame_t.shape[0]))
                        
                        masked_frame = cv.bitwise_or(masked_frame, np.where(rot_img == 255, 255, masked_frame_t))
                        continue

                    else:
                        # Calculate the slope of the connecting line & angle to the horizontal
                        p1 = landmark_screen_coords[min_x_lm]
                        p2 = landmark_screen_coords[max_x_lm]
                        slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))
                        prev_slope = slope
                        rot_angle = compute_rot_angle(slope1=slope, slope2=prev_slope)
                        angle_from_x_axis = compute_rot_angle(slope1=prev_slope)

                        # Compute the center bisecting line of the landmark
                        cx = round((p2.get('y') + p1.get('y'))/2)
                        cy = round((p2.get('x') + p1.get('x'))/2)
                        
                        # Generate the rectangle
                        rectangle = cv.rectangle(masked_frame, (p1.get('x')-50, cx - 50), (p2.get('x') + 50, cx + 50), (255,255,255), -1)
                        masked_frame_t = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        
                        # Generate rotation matrix and rotate the rectangle
                        rot_mat = cv.getRotationMatrix2D((cy,cx), (rot_angle + angle_from_x_axis), 1)
                        rot_img = cv.warpAffine(rectangle, rot_mat, (masked_frame_t.shape[1], masked_frame_t.shape[0]))
                        
                        masked_frame = cv.bitwise_or(masked_frame, np.where(rot_img == 255, 255, masked_frame_t))
                        continue
                else:    
                    mask = get_mask_from_path(frame, landmark_path, face_mesh)
                    masked_frame = cv.bitwise_or(masked_frame, mask)
            
            frame = occlude_frame(frame, masked_frame, occlusion_fill, landmark_screen_coords)

            if static_image_mode:
                success = cv.imwrite(output_dir + "\\" + filename + "_occluded" + extension, frame)
                if not success:
                    logger.error("Function encountered an FileWriteError attempting to call cv2.imwrite(). ")
                    debug_logger.error("Function encountered an FileWriteError while attempting to call cv2.imwrite(). " 
                                      f"Ensure output_dir path string is valid, and ensure {file} is not corrupt.")
                    raise FileWriteError()
                else:
                    break
            else:
                result.write(frame)

        if not static_image_mode:
            capture.release()
            result.release()
        
        logger.info(f"Function execution completed successfully. View outputted file(s) at {output_dir}.")