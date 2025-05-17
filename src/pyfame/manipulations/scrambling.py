from pyfame.utils.predefined_constants import *
from pyfame.mesh.landmarks import *
from pyfame.utils.utils import get_variable_name, compute_rot_angle
from pyfame.utils.exceptions import *
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
from skimage.util import *
from operator import itemgetter
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def facial_scramble(input_dir:str, output_dir:str, out_grayscale:bool = False, scramble_method:int = HIGH_LEVEL_GRID_SCRAMBLE, rand_seed:int|None = None, grid_scramble_threshold:int = 2,
                    grid_square_size:int = 40, with_sub_dirs:bool = False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    """ For each input photo or video file, randomly shuffles the face/facial-landmarks. Output files can either be in full
    color, or grayscale. Specify a random seed for reproducable results.

    Parameters
    ----------

    input_dir: str
        A path string to a directory containing the video files to be processed.

    output_dir: str
        A path string to a directory where processed image or video files will be written too.

    out_grayscale: bool
        A boolean flag indicating if the output file should, or should not be converted to grayscale.

    scramble_method: int
        One of LOW_LEVEL_GRID_SCRAMBLE, HIGH_LEVEL_GRID_SCRAMBLE or LANDMARK_SCRAMBLE. 
    
    rand_seed: int
        An integer used to seed the random number generator used in scrambling. 
    
    grid_scramble_threshold: int
        The maximum number of grid positions that a particular grid square can move. When the image grid is randomly scrambled, 
        each grid square can move at most *threshold* positions in the x and y axis.
    
    grid_square_size: int
        The square dimensions of each grid square used in RANDOM_GRID_SCRAMBLE. The default value of 40, represents grid
        squares of size 40 pixels by 40 pixels.

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

    TypeError:
        Given invalid input parameter typings.
    ValueError: 
        Given an unspecified shuffle_method.
    OSError:
        Given invalid pathstrings to input files. 
    """

    static_image_mode = False
    single_file = False

    # Performing checks on input parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Facial_scramble: parameter input_dir expects a string.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid system path string.")
        raise OSError("Facial_scramble: input_dir must be a valid pathstring to a file or directory.")
    elif os.path.isfile(input_dir):
        single_file = True

    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Facial_scramble: parameter output_dir expects a string.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid system path string.")
        raise OSError("Facial_scramble: output_dir must be a valid pathstring to a directory.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered an ValueError for input parameter output_dir. "
                       "Message: output_dir is not a valid system path to a directory.")
        raise ValueError("Facial_scramble: output_dir must be a valid pathstring to a directory.")
    
    if not isinstance(out_grayscale, bool):
        logger.warning("Function encountered a TypeError for input parameter out_grayscale. "
                       "Message: invalid type for parameter out_grayscale, expected bool.")
        raise TypeError("Facial_scramble: parameter out_grayscale expects a boolean.")
    
    if not isinstance(scramble_method, int):
        logger.warning("Function encountered a TypeError for input parameter scramble_method. "
                       "Message: invalid type for parameter scramble_method, expected int.")
        raise TypeError("Facial_scramble: parameter shuffle_method expects an integer.")
    elif scramble_method not in [27, 28, 29]:
        logger.warning("Function encountered a ValueError for input parameter scramble_method. "
                       "Message: unrecognized value for parameter scramble_method, please see pyfame_utils.display_scramble_method_options() " 
                       "for the full list of accepted values.")
        raise ValueError("Facial_scramble: Unrecognized value for parameter scramble_method "
                         "Please see pyfame_utils.display_scramble_method_options() for the full list of accepted values.")
    
    if rand_seed != None:
        if not isinstance(rand_seed, int):
            logger.warning("Function encountered a TypeError for input parameter rand_seed. "
                           "Message: invalid type for parameter rand_seed, expected int.")
            raise TypeError("Facial_scramble: parameter rand_seed expects an integer.")
    
    if not isinstance(grid_scramble_threshold, int):
        logger.warning("Function encountered a TypeError for input parameter grid_scramble_threshold. "
                       "Message: invalid type for parameter grid_scramble_threshold, expected int.")
        raise TypeError("Facial_scramble: parameter grid_scramble_threshold expects an integer")
    
    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Facial_scramble: parameter with_sub_dirs expects a boolean.")
    
    if not isinstance(min_detection_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_detection_confidence. "
                       "Message: invalid type for parameter min_detection_confidence, expected float.")
        raise TypeError("Facial_scramble: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_detection_confidence. "
                       "Message: min_detection_confidence must be a float in the range [0,1].")
        raise ValueError("Facial_scramble: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_tracking_confidence. "
                       "Message: invalid type for parameter min_tracking_confidence, expected float.")
        raise TypeError("Facial_scramble: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_tracking_confidence. "
                       "Message: min_tracking_confidence must be a float in the range [0,1].")
        raise ValueError("Facial_scramble: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Logging input parameters
    logger.info("Now entering function facial_scramble().")

    scramble_method_name = get_variable_name(scramble_method, globals())
    logger.info(f"Input parameters: out_grayscale = {out_grayscale}, scramble_method = {scramble_method_name}, rand_seed = {rand_seed}, "
                f"grid_scramble_threshold = {grid_scramble_threshold}, grid_square_size = {grid_square_size}, with_sub_dirs = {with_sub_dirs}.")
    
    logger.info(f"Mediapipe configurations: min_detection_confidence = {min_detection_confidence}, min_tracking_confidence = {min_tracking_confidence}.")
    
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
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
    if not os.path.isdir(output_dir + "\\Scrambled"):
        os.mkdir(output_dir + "\\Scrambled")
        output_dir = output_dir + "\\Scrambled"
        logger.info(f"Function created new output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\Scrambled"

    for file in files_to_process:
            
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        frame = None
        shuffled_keys = None
        rot_angles = None
        x_displacements = None
        rng = None
        dir_file_path = output_dir + f"\\{filename}_scrambled{extension}"

        if rand_seed != None:
            rng = np.random.default_rng(rand_seed)
        else:
            rng = np.random.default_rng()

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
                logger.error("Function has encountered an unparseable file type, Function exiting with status 1. " 
                             "Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                    "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                raise UnrecognizedExtensionError()

        # Initialise videoCapture and videoWriter objects
        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoCapture() object. "
                            "Function exiting with status 1.")
                debug_logger.error("Function has encounterd an error attempting to initialize cv2.VideoCapture() object "
                                f"with file {file}. The file may be corrupt or encoded into an unparseable file type.")
                raise FileReadError()
            
            size = (int(capture.get(3)), int(capture.get(4)))

            if out_grayscale == True:
                result = cv.VideoWriter(output_dir + "\\" + filename + "_scrambled" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size, isColor=False)
            else:
                result = cv.VideoWriter(output_dir + "\\" + filename + "_scrambled" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size, isColor=True)
                
            if not result.isOpened():
                logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                            "Function exiting with status 1.")
                debug_logger.error("Function has encountered an error attempting to initialize cv2.VideoWriter() object. "
                                f"Please ensure that {dir_file_path} is a valid path in your current working directory tree.")
                raise FileWriteError()
            
            success, frame = capture.read()
            if not success:
                logger.error(f"Function has encountered an error attempting to read a frame, file may be corrupt. "
                             "Function exiting with status 1.")
                debug_logger.error("Function has encountered an error attempting to read in a frame from file "
                                   f"{file}. file may be corrupt or incorrectly encoded.")
                raise FileReadError()
        else:
            frame = cv.imread(file)
            if frame is None:
                logger.error(f"Function has encountered an error attempting to read a frame, file may be corrupt")
                debug_logger.error("Function has encountered an error attempting to read in a frame from file "
                                   f"{file}. file may be corrupt or incorrectly encoded.")
                raise FileReadError()

        if scramble_method != LANDMARK_SCRAMBLE:
            # Precomputing shuffled grid positions
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

            fo_screen_coords = []

            # Face oval screen coordinates
            for cur_source, cur_target in FACE_OVAL_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                fo_screen_coords.append((source.get('x'), source.get('y')))
                fo_screen_coords.append((target.get('x'), target.get('y')))
            
            fo_mask = np.zeros((frame.shape[0], frame.shape[1]))
            fo_mask = cv.fillConvexPoly(fo_mask, np.array(fo_screen_coords), 1)
            fo_mask = fo_mask.astype(bool)

            # Get x and y bounds of the face oval
            max_x = max(fo_screen_coords, key=itemgetter(0))[0]
            min_x = min(fo_screen_coords, key=itemgetter(0))[0]

            max_y = max(fo_screen_coords, key=itemgetter(1))[1]
            min_y = min(fo_screen_coords, key=itemgetter(1))[1]

            height = max_y-min_y
            width = max_x-min_x

            # Calculate x and y padding, ensuring integer
            x_pad = grid_square_size - (width % grid_square_size)
            y_pad = grid_square_size - (height % grid_square_size)

            if x_pad % 2 !=0:
                min_x -= np.floor(x_pad/2)
                max_x += np.ceil(x_pad/2)
            else:
                min_x -= x_pad/2
                max_x += x_pad/2
            
            if y_pad % 2 !=0:
                min_y -= np.floor(y_pad/2)
                max_y += np.ceil(y_pad/2)
            else:
                min_y -= y_pad/2
                max_y += y_pad/2
                
            # Ensure integer
            min_x = int(min_x)
            max_x = int(max_x)
            min_y = int(min_y)
            max_y = int(max_y)

            height = max_y-min_y
            width = max_x-min_x
            cols = int(width/grid_square_size)
            rows = int(height/grid_square_size)

            grid_squares = {}

            # Populate the grid_squares dict with segments of the frame
            for i in range(rows):
                for j in range(cols):
                    grid_squares.update({(i,j):frame[min_y:min_y + grid_square_size, min_x:min_x + grid_square_size]})
                    min_x += grid_square_size
                min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
                min_y += grid_square_size
            
            keys = list(grid_squares.keys())

            # Shuffle the keys of the grid_squares dict
            if scramble_method == LOW_LEVEL_GRID_SCRAMBLE:
                shuffled_keys = keys.copy()
                shuffled_keys = np.array(shuffled_keys, dtype="i,i")
                shuffled_keys = shuffled_keys.reshape((rows, cols))

                ref_keys = keys.copy()
                ref_keys = np.array(ref_keys, dtype="i,i")
                ref_keys = ref_keys.reshape((rows, cols))

                visited_keys = set()

                # Localised threshold based shuffling of the grid
                for y in range(rows):
                    for x in range(cols):
                        if (x,y) in visited_keys:
                            continue
                        else:
                            x_min = max(0, x - grid_scramble_threshold)
                            x_max = min(cols-1, x + grid_scramble_threshold)
                            y_min = max(0, y - grid_scramble_threshold)
                            y_max = min(rows - 1, y + grid_scramble_threshold)

                            valid_new_positions = [
                                (new_x, new_y)
                                for new_x in range(x_min, x_max + 1)
                                for new_y in range(y_min, y_max + 1)
                                if (new_x, new_y) not in visited_keys
                            ]

                            if valid_new_positions:
                                new_x, new_y = rng.choice(valid_new_positions)

                                # Perform the positional swap
                                shuffled_keys[new_y,new_x] = ref_keys[y,x]
                                shuffled_keys[y,x] = ref_keys[new_y, new_x]
                                visited_keys.add((new_x, new_y))
                                visited_keys.add((x,y))
                            else:
                                visited_keys.add((x,y))

                shuffled_keys = list(shuffled_keys.reshape((-1,)))
                # Ensure tuple(int) 
                shuffled_keys = [tuple([int(x), int(y)]) for (x,y) in shuffled_keys]
            else:
                # Scramble the keys of the grid_squares dict
                shuffled_keys = keys.copy()
                rng.shuffle(shuffled_keys)
        else:
                rot_angles = {}
                x_displacements = {}

                for i in range(4):
                    rn = rng.random()

                    if i+1 < 3:
                            if rn < 0.25:
                                rot_angles.update({i+1:90})
                            elif rn < 0.5:
                                rot_angles.update({i+1:-90})
                            elif rn < 0.75:
                                rot_angles.update({i+1:180})
                            else:
                                rot_angles.update({i+1:0})
                    elif i+1 == 3:
                        if rn < 0.5:
                            rot_angles.update({i+1:90})
                        else:
                            rot_angles.update({i+1:-90})
                    else:
                        if rn < 0.5:
                            rot_angles.update({i+1:180})
                        else:
                            rot_angles.update({i+1:0})
                    
                    if rn < 0.5:
                        x_displacements.update({i+1:int(-40 * rng.random())})
                    else:
                        x_displacements.update({i+1:int(40 * rng.random())})

        # Main Processing loop for video files (will only iterate once over images)
        while True:
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
            
            output_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            
            match scramble_method:
                case 27 | 28: # Grid scrambling
                    fo_screen_coords = []

                    # Face oval screen coordinates
                    for cur_source, cur_target in FACE_OVAL_PATH:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        fo_screen_coords.append((source.get('x'), source.get('y')))
                        fo_screen_coords.append((target.get('x'), target.get('y')))
                    
                    fo_mask = np.zeros((frame.shape[0], frame.shape[1]))
                    fo_mask = cv.fillConvexPoly(fo_mask, np.array(fo_screen_coords), 1)
                    fo_mask = fo_mask.astype(bool)
                    
                    min_x = min(fo_screen_coords, key=itemgetter(0))[0]
                    min_y = min(fo_screen_coords, key=itemgetter(1))[1]

                    grid_squares = {}

                    # Populate the grid_squares dict with segments of the frame
                    for i in range(rows):
                        for j in range(cols):
                            grid_squares.update({(i,j):frame[min_y:min_y + grid_square_size, min_x:min_x + grid_square_size]})
                            min_x += grid_square_size
                        min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
                        min_y += grid_square_size
                    
                    min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
                    min_y = int(min(fo_screen_coords, key=itemgetter(1))[1])
                    
                    keys = list(grid_squares.keys())

                    # Populate the scrambled grid dict
                    shuffled_grid_squares = {}

                    for old_key, new_key in zip(keys, shuffled_keys):
                        square = grid_squares.get(new_key)
                        shuffled_grid_squares.update({old_key:square})

                    # Fill the output frame with scrambled grid segments
                    for i in range(rows):
                        for j in range(cols):
                            cur_square = shuffled_grid_squares.get((i,j))
                            output_frame[min_y:min_y+grid_square_size, min_x:min_x+grid_square_size] = cur_square
                            min_x += grid_square_size
                        min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
                        min_y += grid_square_size
                    
                    min_x = int(min(fo_screen_coords, key=itemgetter(0))[0])
                    min_y = int(min(fo_screen_coords, key=itemgetter(1))[1])

                    # Calculate the slope of the connecting line & angle to the horizontal
                    # landmarks 162, 389 form a paralell line to the x-axis when the face is vertical
                    p1 = landmark_screen_coords[162]
                    p2 = landmark_screen_coords[389]
                    slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))

                    if slope > 0:
                        angle_from_x_axis = (-1)*compute_rot_angle(slope1=slope)
                    else:
                        angle_from_x_axis = compute_rot_angle(slope1=slope)
                    cx = min_x + (width/2)
                    cy = min_y + (height/2)

                    # Using the rotation angle, generate a rotation matrix and apply affine transform
                    rot_mat = cv.getRotationMatrix2D((cx,cy), (angle_from_x_axis), 1)
                    output_frame = cv.warpAffine(output_frame, rot_mat, (frame.shape[1], frame.shape[0]))

                    # Ensure grid only overlays the face oval
                    masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                    masked_frame[fo_mask] = 255
                    output_frame = np.where(masked_frame == 255, output_frame, frame)
                    output_frame = output_frame.astype(np.uint8)

                case 29: # Landmark scrambling
                    le_screen_coords = []
                    re_screen_coords = []
                    nose_screen_coords = []
                    lips_screen_coords = []
                    fo_screen_coords = []
                    
                    # Left eye screen coordinates
                    for cur_source, cur_target in LEFT_EYE_PATH:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        le_screen_coords.append((source.get('x'), source.get('y')))
                        le_screen_coords.append((target.get('x'), target.get('y')))
                    
                    # Right eye screen coordinates
                    for cur_source, cur_target in RIGHT_EYE_PATH:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        re_screen_coords.append((source.get('x'), source.get('y')))
                        re_screen_coords.append((target.get('x'), target.get('y')))
                    
                    # Nose screen coordinates
                    for cur_source, cur_target in NOSE_PATH:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        nose_screen_coords.append((source.get('x'), source.get('y')))
                        nose_screen_coords.append((target.get('x'), target.get('y')))
                    
                    # Lips screen coordinates
                    for cur_source, cur_target in MOUTH_PATH:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        lips_screen_coords.append((source.get('x'), source.get('y')))
                        lips_screen_coords.append((target.get('x'), target.get('y')))
                    
                    # Face oval screen coordinates
                    for cur_source, cur_target in FACE_OVAL_TIGHT_PATH:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        fo_screen_coords.append((source.get('x'), source.get('y')))
                        fo_screen_coords.append((target.get('x'), target.get('y')))

                    # Creating boolean masks for the facial landmark regions
                    le_mask = np.zeros((frame.shape[0],frame.shape[1]))
                    le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                    le_mask = le_mask.astype(bool)

                    re_mask = np.zeros((frame.shape[0],frame.shape[1]))
                    re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                    re_mask = re_mask.astype(bool)

                    nose_mask = np.zeros((frame.shape[0], frame.shape[1]))
                    nose_mask = cv.fillConvexPoly(nose_mask, np.array(nose_screen_coords), 1)
                    nose_mask = nose_mask.astype(bool)

                    lip_mask = np.zeros((frame.shape[0],frame.shape[1]))
                    lip_mask = cv.fillConvexPoly(lip_mask, np.array(lips_screen_coords), 1)
                    lip_mask = lip_mask.astype(bool)

                    fo_mask = np.zeros((frame.shape[0], frame.shape[1]))
                    fo_mask = cv.fillConvexPoly(fo_mask, np.array(fo_screen_coords), 1)
                    fo_mask = fo_mask.astype(bool)

                    masks = [le_mask, re_mask, nose_mask, lip_mask]
                    screen_coords = [le_screen_coords, re_screen_coords, nose_screen_coords, lips_screen_coords]
                    lms = []
                    output_frame = frame.copy()

                    # Cut out, and store landmarks
                    for mask, coords in zip(masks, screen_coords):
                        im_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        im_mask[mask] = 255

                        max_x = max(coords, key=itemgetter(0))[0]
                        min_x = min(coords, key=itemgetter(0))[0]

                        max_y = max(coords, key=itemgetter(1))[1]
                        min_y = min(coords, key=itemgetter(1))[1]

                        # Compute the center bisecting lines of the landmark
                        cx = round((max_y + min_y)/2)           
                        cy = round((max_x + min_x)/2)

                        # Cut out landmark region and store it
                        lm = cv.bitwise_and(src1=frame, src2=frame, mask=im_mask)
                        lms.append((lm, (cy,cx)))

                        # Fill landmark holes with inpainting
                        output_frame[mask] = 0
                        output_frame = cv.inpaint(output_frame, im_mask, 10, cv.INPAINT_NS)

                    landmarks = dict(map(lambda i,j: (i,j), [1,2,3,4], lms))

                    for key in landmarks:
                        # Get the landmark, and the center point of its position
                        landmark, center = landmarks[key]
                        cx, cy = center
                        h,w = landmark.shape[:2]

                        rot_angle = rot_angles[key]
                        x_disp = x_displacements[key]

                        # Generate rotation matrices for the landmark
                        if key == 3:
                            rot_mat = cv.getRotationMatrix2D(center=center, angle=rot_angle, scale=1)
                            landmark = cv.warpAffine(landmark, rot_mat, (w,h))
                            cy += 20
                        else:
                            rot_mat = cv.getRotationMatrix2D(center=center, angle=rot_angle, scale=1)
                            landmark = cv.warpAffine(landmark, rot_mat, (w,h))
                        
                        cx += x_disp

                        # Create landmark mask
                        lm_mask = np.zeros((landmark.shape[0], landmark.shape[1]), dtype=np.uint8)
                        lm_mask = np.where(landmark != 0, 255, 0)
                        lm_mask = lm_mask.astype(np.uint8)
                        
                        # Clone the landmark onto the original face in its new position
                        output_frame = cv.seamlessClone(landmark, output_frame, lm_mask, (cx, cy), cv.NORMAL_CLONE)
                case _:
                    logger.error("Function has encountered an unrecognized value for parameter scramble_method during execution. "
                                 "Function exiting with status 1.")
                    debug_logger.error("Function has encountered an unrecognized value for parameter scramble_method during execution. "
                                       "Input parameter checks may not be performing as expected.")
                    raise ValueError("Unrecognized value for parameter scramble_method.")

            if not static_image_mode:
                if out_grayscale == True:
                    output_frame = cv.cvtColor(output_frame, cv.COLOR_BGR2GRAY)
                
                result.write(output_frame)
                success, frame = capture.read()
                if not success:
                    break 
            else:
                if out_grayscale == True:
                    output_frame = cv.cvtColor(output_frame, cv.COLOR_BGR2GRAY)

                success = cv.imwrite(output_dir + "\\" + filename + "_scrambled" + extension, output_frame)
                if not success:
                    logger.error("Function has encountered an error attempting to call cv2.imwrite(), exiting with status 1.")
                    debug_logger.error("Function has encountered an error attempting to call cv2.imwrite() to directory "
                                       f"{dir_file_path}. Please ensure that this path is a valid path in your current working directory tree.")
                    raise FileWriteError()
                break
        
        if not static_image_mode:
            capture.release()
            result.release()
        
        logger.info(f"Function execution completed successfully, view outputted file(s) at {output_dir}.")
