from pyfame.util.util_exceptions import *
from pyfame.mesh import *
from pyfame.io import *
from pyfame.util import compute_rot_angle
import cv2 as cv
import numpy as np
import os
import mediapipe as mp
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def apply_image_overlay(input_dir:str, output_dir:str, overlay_object:int, overlay_center_pos:tuple[int] = None, 
                  with_sub_dirs:bool = False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    
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

    # Defining the mediapipe facemesh task
    face_mesh = None

    # Creating a list of file names to iterate through when processing
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
    
    logger.info(f"Function has read {len(files_to_process)} file(s) from input directory {input_dir}.")
    
    output_dir = create_output_directory(output_dir, "\\Image_Overlayed")

    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        dir_file_path = output_dir + f"\\{filename}_overlayed{extension}"
        codec = None

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
                logger.error("Function has encountered an unparseable file type. " 
                             "Function exiting with status 1. Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                    "Consider using different input file formats, or transcoding video files with pyfameutils.transcode_video_to_mp4().")
                raise UnrecognizedExtensionError()

        capture = None
        result = None
        prev_slope = None

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
            
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            landmark_screen_coords = get_mesh_coordinates(frame_rgb, face_mesh)
            
            face_oval_coords = get_mesh_coordinates_from_path(frame_rgb, face_mesh, FACE_OVAL_PATH)
            
            # Get the facial width to scale the overlayed object
            max_x = max(x for x,_ in face_oval_coords)
            min_x = min(x for x,_ in face_oval_coords)

            if overlay_object == 1:

                # Read in sunglasses image
                sunglasses = cv.imread("src//pyfame//sample_data//overlay_images//sunglasses.png", cv.IMREAD_UNCHANGED)
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