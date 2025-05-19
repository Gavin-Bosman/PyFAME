from pyfame.utils.exceptions import *
from pyfame.mesh.landmarks import *
from pyfame.utils import compute_rot_angle
import cv2 as cv
import numpy as np
import os
import mediapipe as mp
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def overlay_image(input_dir:str, output_dir:str, overlay_object:int, overlay_center_pos:tuple[int] = None, 
                  with_sub_dirs:bool = False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    single_file = False
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
    elif os.path.isfile(input_dir):
        single_file = True
    
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

    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)

    # Creating a list of file names to iterate through when processing
    files_to_process = []
    if single_file:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    logger.info(f"Function has read {len(files_to_process)} file(s) from input directory {input_dir}.")
    
    if not os.path.exists(output_dir + "\\Image_Overlayed"):
        os.mkdir(output_dir + "\\Image_Overlayed")
        output_dir = output_dir + "\\Image_Overlayed"
        logger.info(f"Created output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\Image_Overlayed"

    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        dir_file_path = output_dir + f"\\{filename}_overlayed{extension}"
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
        prev_slope = None

        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                logger.error("Function encountered an error while attempting to initialize cv2.VideoCapture() object. "
                             "Function exiting with status 1.")
                debug_logger.error("Function encountered an error while attempting instantiating cv2.VideoCapture() object. "
                                   f"File {file} may be corrupt or encoded in an invalid format.")
                raise FileReadError()

            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_overlayed" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                logger.error("Function encountered an error while attempting to initialize cv2.VideoWriter() object. "
                             "Function exiting with status 1.")
                debug_logger.error(f"Function encountered an error while attempting to initialize cv2.VideoWriter() object. "
                                   f"Check that {dir_file_path} is a valid path to a file in your system.")
                raise FileWriteError()

        while True:

            if static_image_mode:
                frame = cv.imread(file)
                if frame is None:
                    raise FileReadError()
            else:
                success, frame = capture.read()
                if not success:
                    break    
            
            #frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)

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
            
            face_oval_coords = []
            # Converting landmark coords to screen coords
            for cur_source, cur_target in FACE_OVAL_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                face_oval_coords.append((source.get('x'),source.get('y')))
                face_oval_coords.append((target.get('x'),target.get('y')))
            
            # Get the facial width to scale the overlayed object
            max_x = max(x for x,_ in face_oval_coords)
            min_x = min(x for x,_ in face_oval_coords)

            if overlay_object == 1:

                # Read in sunglasses image
                sunglasses = cv.imread("src//pyfame//public//overlay_images//sunglasses.png", cv.IMREAD_UNCHANGED)
                if sunglasses is None:
                    raise FileReadError()

                # Rescaling the overlay image
                overlay_width = sunglasses.shape[1]
                overlay_height = sunglasses.shape[0]
                scaling_factor = 1/(overlay_width/(max_x-min_x))
                sunglasses = cv.resize(sunglasses, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)
                overlay_width = sunglasses.shape[1]
                overlay_height = sunglasses.shape[0]

                rot_angle = 0.0
                angle_to_x = 0.0

                # Rotating the overlay 227 447
                if prev_slope is None:
                    p1 = landmark_screen_coords[227]
                    p2 = landmark_screen_coords[447]
                    cur_slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))
                    rot_angle = compute_rot_angle(slope1=cur_slope)
                    prev_slope = cur_slope
                else:
                    p1 = landmark_screen_coords[227]
                    p2 = landmark_screen_coords[447]
                    cur_slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))
                    rot_angle = compute_rot_angle(slope1=cur_slope, slope2=prev_slope)
                    angle_to_x = compute_rot_angle(slope1=prev_slope)
                    prev_slope = cur_slope

                # Add transparent padding prior to rotation
                diag_size = int(np.ceil(np.sqrt(overlay_height**2 + overlay_width**2)))
                pad_h = (diag_size-overlay_height)//2
                pad_w = (diag_size-overlay_width)//2
                padded = np.zeros((diag_size, diag_size, 4), dtype=np.uint8)
                padded[pad_h:pad_h+overlay_height, pad_w:pad_w + overlay_width] = sunglasses

                # Get center point of padded overlay
                padded_height = padded.shape[0]
                padded_width = padded.shape[1]
                padded_center = (overlay_width//2, overlay_height//2)

                # Perform rotation
                if prev_slope is None:
                    rot_mat = cv.getRotationMatrix2D(padded_center, rot_angle, 1)
                else:
                    rot_mat = cv.getRotationMatrix2D(padded_center, (rot_angle + angle_to_x), 1)
                sunglasses = cv.warpAffine(padded, rot_mat, (padded_width, padded_height), flags=cv.INTER_LINEAR)

                overlay_img = sunglasses[:,:,:3]
                overlay_mask = sunglasses[:,:,3] / 255.0
                overlay_mask = overlay_mask[:,:,np.newaxis]
                overlay_width = sunglasses.shape[1]
                overlay_height = sunglasses.shape[0]

                facial_center = landmark_screen_coords[6]
                x_pos = facial_center.get('x') - padded_width//2
                y_pos = facial_center.get('y') - padded_height//2

                roi = frame[y_pos:y_pos + padded_height, x_pos:x_pos + padded_width]
                blended = (1.0 - overlay_mask) * roi + overlay_mask * overlay_img

                frame[y_pos:y_pos + padded_height, x_pos:x_pos + padded_width] = blended.astype(np.uint8)
                
                if static_image_mode:
                    success = cv.imwrite(output_dir + "\\" + filename + "_overlayed" + extension, frame)
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
            
                
