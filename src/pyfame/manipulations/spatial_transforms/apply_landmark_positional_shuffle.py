from pyfame.utils.predefined_constants import *
from pyfame.mesh.landmarks import *
from pyfame.utils.utils import get_variable_name
from pyfame.utils.exceptions import *
from pyfame.io import *
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
from skimage.util import *
from operator import itemgetter
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def apply_landmark_shuffle(input_dir:str, output_dir:str, rand_seed:int | None, out_grayscale:bool = False, with_sub_dirs:bool = False,
                           min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    static_image_mode = False

    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
    
    logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")

    # Creating named output directories for video output
    output_dir = create_output_dir(output_dir, "Landmark_Shuffled")

    for file in files_to_process:
            
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        frame = None
        rot_angles = None
        x_displacements = None
        rng = None
        dir_file_path = output_dir + f"\\{filename}_grid_shuffled{extension}"

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
            capture = get_video_capture(file)
            size = (int(capture.get(3)), int(capture.get(4)))
            result = get_video_writer(dir_file_path, size, codec=codec, isColor=(not out_grayscale))
            
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
