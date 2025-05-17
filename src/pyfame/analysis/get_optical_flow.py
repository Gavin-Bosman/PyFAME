from pyfame.utils.predefined_constants import *
from pyfame.mesh.landmarks import *
from pyfame.io import *
from pyfame.utils.utils import get_variable_name
from pyfame.utils.exceptions import *
from pyfame.manipulations.occlusion import get_mask_from_path
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
from skimage.util import *
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def get_optical_flow(input_dir:str, output_dir:str, optical_flow_type: int|str = SPARSE_OPTICAL_FLOW, landmarks_to_track:list[int]|None = None,
                     max_corners:int = 20, corner_quality_lvl:float = 0.3, min_corner_distance:int = 7, block_size:int = 5, win_size:tuple[int] = (15,15), 
                     max_pyr_lvl:int = 2, pyr_scale:float = 0.5, max_iter:int = 10, lk_accuracy_thresh:float = 0.03, poly_sigma:float = 1.2, 
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
    
    max_iter: int
        The maximum number of iterations (over each frame) the optical flow algorithm will make before terminating.

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

    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Get_optical_flow: parameter input_dir expects a string.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path in your current working tree.")
        raise OSError("Get_optical_flow: parameter input_dir is required to be a valid pathstring.")
    
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
        raise ValueError("Get_optical_flow: parameter output_dir must be a path string to a directory.")
    
    if not isinstance(optical_flow_type, int):
        if not isinstance(optical_flow_type, str):
            logger.warning("Function encountered a TypeError for input parameter optical_flow_type. "
                           "Message: invalid type for parameter optical_flow_type, expected str or int.")
            raise TypeError("Get_optical_flow: parameter optical_flow_type expects a string or integer.")
        elif str.lower(optical_flow_type) not in ["sparse", "dense"]:
            logger.warning("Function encountered a ValueError for input parameter optical_flow_type. "
                           "Message: unrecognized value for parameter optical_flow_type, please see utils."
                           "display_optical_flow_options() for a full list of accepted values.")
            raise ValueError("Get_optical_flow: parameter optical_flow_type must be one of 'sparse' or 'dense'.")
        else:
            if str.lower(optical_flow_type) == "sparse":
                optical_flow_type = SPARSE_OPTICAL_FLOW
            elif str.lower(optical_flow_type) == "dense":
                optical_flow_type = DENSE_OPTICAL_FLOW
    elif optical_flow_type not in [SPARSE_OPTICAL_FLOW, DENSE_OPTICAL_FLOW]:
        logger.warning("Function encountered a ValueError for input parameter optical_flow_type. "
                       "Message: unrecognized value for parameter optical_flow_type, please see utils."
                       "display_optical_flow_options() for a full list of accepted values.")
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
                raise ValueError("Get_optical_flow: parameter landmarks_to_track must be a list of integers.")
    
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
        logger.warning("Function encountered a ValueError for input parameter win_size. "
                       "Message: unrecognized value of parameter win_size, expected tuple[int].")
        raise ValueError("Get_optical_flow: parameter win_size must be a tuple of integers.")
    
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
    
    if not isinstance(max_iter, int):
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
            logger.warning("Function encountered a ValueError for input parameter point_color. "
                           "Message: Unrecognized value of parameter point_color, expected tuple[int].")
            raise ValueError("Get_optical_flow: parameter point color must be a tuple of integers.")
    
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
    logger.info(f"Input parameters continued... max_pyr_level = {max_pyr_lvl}, pyr_scale = {pyr_scale}, max_lk_iter = {max_iter}, lk_accuracy_thresh = {lk_accuracy_thresh}, "
                f"poly_sigma = {poly_sigma}, point_color = {point_color}, point_radius = {point_radius}, vector_color = {vector_color}, with_sub_dirs = {with_sub_dirs}.")
    logger.info(f"Mediapipe configurations: min_detection_confidence = {min_detection_confidence}, min_tracking_confidence = {min_tracking_confidence}.")
    
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
    
    logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")
    
    # Creating named output directories for video output
    output_dir = create_output_dir(output_dir, "Optical_Flow")

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
        
        # Instantiating video read/writers
        capture = get_video_capture(file)
        size = (int(capture.get(3)), int(capture.get(4)))
        result = get_video_writer(dir_file_path, size, codec=codec)
        
        if optical_flow_type == SPARSE_OPTICAL_FLOW:
            csv = open(output_dir + "\\" + filename + "_optical_flow.csv", "w")
            csv.write("Timestamp,X_old,Y_old,X_new,Y_new,Magnitude,Angle,Status,Error\n")
        elif optical_flow_type == DENSE_OPTICAL_FLOW:
            csv = open(output_dir + "\\" + filename + "_optical_flow.csv", "w")
            csv.write("Timestamp,dx,dy,Magnitude,Angle\n")

        # Defining persistent loop params
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
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, max_iter, lk_accuracy_thresh))

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
                logger.error("Face mesh detection error, function exiting with status 1.")
                debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process() on the current frame.")
                raise FaceNotFoundError()
            
            # Create face oval image mask
            face_mask = get_mask_from_path(frame, FACE_OVAL_PATH, face_mesh)

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
                    flow = cv.calcOpticalFlowFarneback(old_gray, gray_frame, None, pyr_scale, max_pyr_lvl, win_size[0], max_iter, block_size, poly_sigma, 0)

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