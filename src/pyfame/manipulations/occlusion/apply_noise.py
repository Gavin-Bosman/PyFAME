from pyfame.utils.predefined_constants import *
from pyfame.mesh.landmarks import *
from pyfame.utils.utils import get_variable_name
from pyfame.utils.exceptions import *
from pyfame.io import *
from .apply_mask import mask_frame
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
from skimage.util import *
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def blur_face_region(input_dir:str, output_dir:str, blur_method:str | int = "gaussian", mask_type:int = FACE_OVAL_MASK, 
                     k_size:int = 15, with_sub_dirs:bool = False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    """ For each video or image file within `input_dir`, the specified `blur_method` will be applied. Blurred images and video files
    are then written out to `output_dir`.

    Parameters
    ----------

    input_dir: str
        A path string to a directory containing the image or video files to be processed.

    output_dir: str
        A path string to a directory where processed files will be written.

    blur_method: str, int
        Either a string literal ("average", "gaussian", "median"), or a predefined integer constant 
        (BLUR_METHOD_AVERAGE, BLUR_METHOD_GAUSSIAN, BLUR_METHOD_MEDIAN) specifying the type of blurring operation to be performed.
    
    mask_type: int
        An integer flag specifying the type of mask to be applied.
        
    k_size: int
        Specifies the size of the square kernel used in blurring operations. 
    
    with_sub_dirs: bool
        A boolean flag indicating if the input directory contains nested sub-directories.
    
    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.

    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
    
    Raises
    ------

    TypeError:
        Given invalid or incompatible input parameter types.
    ValueError:
        Given an unrecognized value.
    OSError:
        Given invalid path strings to input or output directory.

    """
    static_image_mode = False

    # Performing checks on function parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Blur_face_region: invalid type for parameter input_dir.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path to a directory, or the directory does not exist.")
        raise OSError("Blur_face_region: input directory path is not a valid path, or the directory does not exist.")
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Blur_face_region: parameter output_dir must be a str.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path to a directory, or the directory does not exist.")
        raise OSError("Blur_face_region: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered a ValueError for input parameter output_dir. "
                       "Message: output_dir must be a valid path to a directory.")
        raise ValueError("Blur_face_region: output_dir must be a valid path to a directory.")
    
    if isinstance(blur_method, str):
        if str.lower(blur_method) not in ["average", "gaussian", "median"]:
            logger.warning("Function encountered a ValueError for input parameter blur_method. "
                           "Message: unrecognized value for parameter blur_method.")
            raise ValueError("Blur_face_region: Unrecognised value for parameter blur_method.")
    elif isinstance(blur_method, int):
        if blur_method not in [BLUR_METHOD_AVERAGE, BLUR_METHOD_GAUSSIAN, BLUR_METHOD_MEDIAN]:
            logger.warning("Function encountered a ValueError for input parameter blur_method. "
                           "Message: unrecognized value for parameter blur_method.")
            raise ValueError("Blur_face_region: Unrecognised value for parameter blur_method.")
    else:
        logger.warning("Function encountered a TypeError for input parameter blur_method. "
                       "Message: Invalid type for parameter blur_method, expected int or str.")
        raise TypeError("Blur_face_region: Incompatable type for parameter blur_method.")
    
    if not isinstance(mask_type, int):
        logger.warning("Function encountered a TypeError for input parameter mask_type. "
                       "Message: invalid type for parameter mask_type, expected int.")
        raise TypeError("Blur_face_region: parameter mask_type must be of type int.")
    elif mask_type not in MASK_OPTIONS:
        logger.warning("Function encountered a ValueError for input parameter mask_type. "
                       "Message: unrecognized value for parameter mask_type.")
        raise ValueError("Blur_face_region: unrecognized value for parameter mask_type.")
    
    if not isinstance(k_size, int):
        logger.warning("Function encountered a TypeError for input parameter k_size. "
                       "Message: invalid type for parameter k_size, expected int.")
        raise TypeError("Blur_face_region: parameter k_size must be of type int.")
    elif k_size < 1:
        logger.warning("Function encountered a ValueError for input parameter k_size. "
                       "Message: parameter k_size must be a positive integer (>0).")
        raise ValueError("Blur_face_region: parameter k_size must be a positive integer.")
    
    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Blur_face_region: parameter with_sub_dirs must be of type bool.")
    
    if not isinstance(min_detection_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_detection_confidence. "
                       "Message: invalid type for parameter min_detection_confidence, expected float.")
        raise TypeError("Blur_face_region: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_detection_confidence. "
                       "Message: min_detection_confidence must be a float value in the range [0,1].")
        raise ValueError("Blur_face_region: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_tracking_confidence. "
                       "Message: invalid type for parameter min_tracking_confidence, expected float.")
        raise TypeError("Blur_face_region: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_tracking_confidence. "
                       "Message: min_tracking_confidence must be a float value in the range [0,1].")
        raise ValueError("Blur_face_region: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Logging input parameters

    logger.info("Now entering function blur_face_region().")

    if isinstance(blur_method, str):
        logger.info(f"Input parameters: blur_method = {blur_method}, k_size = {k_size}.")
    else:
        blur_method_name = get_variable_name(blur_method, globals())
        logger.info(f"Input parameters: blur_method = {blur_method_name}, k_size = {k_size}.")

    logger.info(f"Mediapipe configurations: min_detection_confidence = {min_detection_confidence}, "
                f"min_tracking_confidence = {min_tracking_confidence}.")

    
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)

    # Creating a list of file names to iterate through when processing
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
        
    logger.info(f"Function has read {len(files_to_process)} file(s) from input directory {input_dir}.")
    
    output_dir = create_output_dir(output_dir, "Blurred")

    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        dir_file_path = output_dir + f"\\{filename}_blurred{extension}"

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

            masked_frame = mask_frame(frame, face_mesh, mask_type, (0,0,0))

            frame_blurred = None

            match blur_method:
                case "average" | "Average":
                    frame_blurred = cv.blur(frame, (k_size, k_size))
                    frame = np.where(masked_frame != 255, frame_blurred, frame)
                case 11:
                    frame_blurred = cv.blur(frame, (k_size, k_size))
                    frame = np.where(masked_frame != 255, frame_blurred, frame)
                
                case "gaussian" | "Gaussian":
                    frame_blurred = cv.GaussianBlur(frame, (k_size, k_size), 0)
                    frame = np.where(masked_frame != 255, frame_blurred, frame)
                case 12:
                    frame_blurred = cv.GaussianBlur(frame, (k_size, k_size), 0)
                    frame = np.where(masked_frame != 255, frame_blurred, frame)
                
                case "median" | "Median":
                    frame_blurred = cv.medianBlur(frame, k_size)
                    frame = np.where(masked_frame != 255, frame_blurred, frame)
                case 13:
                    frame_blurred = cv.medianBlur(frame, k_size)
                    frame = np.where(masked_frame != 255, frame_blurred, frame)
                
                case _:
                    debug_logger.error("Function encountered a ValueError after parameter checks. "
                                       "Parameter type and value checks may not be performing as intended.")
                    raise ValueError("Unrecognized value for parameter blur_method.")
                    
            if static_image_mode:
                success = cv.imwrite(output_dir + "\\" + filename + "_blurred" + extension, frame)
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

def apply_noise(input_dir:str, output_dir:str, noise_method:str|int = "pixelate", pixel_size:int = 32, noise_prob:float = 0.5,
                rand_seed:int | None = None, mean:float = 0.0, standard_dev:float = 0.5, mask_type:int = FACE_OVAL_MASK, 
                with_sub_dirs:bool = False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    """Takes an input image or video file, and applies the specified noise method to the image or each frame of the video. For
    noise_method `pixelate`, an output image size must be specified in order to resize the image/frame's pixels.

    Parameters
    ----------

    input_dir: str
        A path string to a directory containing the video files to be processed.

    output_dir: str
        A path string to a directory where outputted csv files will be written to.

    noise_method: str or int
        Either an integer flag, or string literal specifying the noise method of choice. For the full list of 
        available options, please see pyfame_utils.display_noise_method_options().

    pixel_size: int
        The pixel scale applied when pixelating the output file.
    
    noise_prob: float
        The probability of noise being applied to a given pixel, default is 0.5.
    
    rand_seed: int or None
        A seed for the random number generator used in gaussian and salt and pepper noise. Allows the user 
        to create reproducable results. 
    
    mean: float
        The mean or center of the gaussian distribution used when sampling for gaussian noise.

    standard_dev: float
        The standard deviation or variance of the gaussian distribution used when sampling for gaussian noise.
    
    mask_type: int or None
        An integer specifying the facial region in which to apply the specified noise operation.

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

    TypeError: Given invalid parameter typings.
    OSError: Given invalid paths for parameters input_dir or output_dir.
    ValueError: Given an unrecognized noise_method or mask_type.

    """

    logger.info("Now entering function apply_noise().")
    static_image_mode = False

    # Type and value checking input parameters
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: invalid type for parameter input_dir, expected str.")
        raise TypeError("Apply_noise: input_dir must be a path string.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path to a directory, or the directory does not exist.")
        raise OSError("Apply_noise: input_dir is not a valid path.")
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: invalid type for parameter output_dir, expected str.")
        raise TypeError("Apply_noise: output_dir must be a path string.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path to a directory, or the directory does not exist.")
        raise OSError("Apply_noise: output_dir is not a valid path.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered an OSError for input parameter output_dir. "
                       "Message: output_dir is not a valid path to a directory, or the directory does not exist.")
        raise ValueError("Apply_noise: output_dir must be a path string to a directory.")
    
    if not isinstance(noise_method, int):
        if not isinstance(noise_method, str):
            logger.warning("Function has encountered a TypeError for input parameter noise_method. " 
                           "Message: invalid type for parameter noise_method, expected int or str.")
            raise TypeError("Apply_noise: parameter noise_method must be either int or str.")
        elif str.lower(noise_method) not in ["salt and pepper", "pixelate", "gaussian"]:
            logger.warning("Function encountered a ValueError for input parameter noise_method. "
                           "Message: unrecognized value for parameter noise_method.")
            raise ValueError("Apply_noise: unrecognized value, please see utils.display_noise_method_options() for the full list of accepted values.")
    elif noise_method not in [NOISE_METHOD_SALT_AND_PEPPER, NOISE_METHOD_PIXELATE, NOISE_METHOD_GAUSSIAN]:
        logger.warning("Function encountered a ValueError for input parameter noise_method. "
                           "Message: unrecognized value for parameter noise_method.")
        raise ValueError("Apply_noise: unrecognized value, please see utils.display_noise_method_options() for the full list of accepted values.")
    
    if not isinstance(pixel_size, int):
        logger.warning("Function encountered a TypeError for input parameter pixel_size. "
                       "Message: invalid type for parameter pixel_size, expected int.")
        raise TypeError("Apply_noise: parameter pixel_size expects an integer.")
    elif pixel_size < 1:
        logger.warning("Function encountered a ValueError for input parameter pixel_size. "
                       "Message: pixel_size must be a positive (>0) integer.")
        raise ValueError("Apply_noise: parameter pixel_size must be a positive integer.")
    
    if not isinstance(noise_prob, float):
        logger.warning("Function encountered a TypeError for input parameter noise_prob. "
                       "Message: invalid type for parameter noise_prob, expected float.")
        raise TypeError("Apply_noise: parameter noise_prob expects a float.")
    elif noise_prob < 0 or noise_prob > 1:
        logger.warning("Function encountered a ValueError for input parameter noise_prob. "
                       "Message: parameter noise_prob must be a float in the range [0,1].")
        raise ValueError("Apply_noise: parameter noise_prob must lie in the range [0,1].")
    
    if rand_seed is not None:
        if not isinstance(rand_seed, int):
            logger.warning("Function encountered a TypeError for input parameter rand_seed. "
                       "Message: invalid type for parameter rand_seed, expected int.")
            raise TypeError("Apply_noise: parameter rand_seed expects an integer.")
    
    if not isinstance(mean, float):
        logger.warning("Function encountered a TypeError for input parameter mean. "
                       "Message: invalid type for parameter mean, expected float.")
        raise TypeError("Apply_noise: parameter mean expects a float.")
    
    if not isinstance(standard_dev, float):
        logger.warning("Function encountered a TypeError for input parameter standard_dev. "
                       "Message: invalid type for parameter standard_dev, expected float.")
        raise TypeError("Apply_noise: parameter standard_dev expects a float.")

    if not isinstance(mask_type, int):
        logger.warning("Function encountered a TypeError for input parameter mask_type. "
                       "Message: invalid type for parameter mask_type, expected int.")
        raise TypeError("Apply_noise: parameter mask_type expects an integer.")
    elif mask_type not in MASK_OPTIONS:
        logger.warning("Function encountered a ValueError for input parameter mask_type. "
                       "Message: unrecognized mask_type. See utils.display_face_mask_options().")
        raise ValueError("Apply_noise: mask_type must be one of the predefined options specified within utils.display_face_mask_options().")
    
    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: invalid type for parameter with_sub_dirs, expected bool.")
        raise TypeError("Apply_noise: parameter with_sub_dirs expects a boolean.")
    
    if not isinstance(min_detection_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_detection_confidence. "
                       "Message: invalid type for parameter min_detection_confidence, expected float.")
        raise TypeError("Apply_noise: parameter min_detection_confidence expects a float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_detection_confidence. "
                       "Message: parameter min_detection_confidence must be a float in the range [0,1].")
        raise ValueError("Apply_noise: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        logger.warning("Function encountered a TypeError for input parameter min_tracking_confidence. "
                       "Message: invalid type for parameter min_tracking_confidence, expected float.")
        raise TypeError("Apply_noise: parameter min_tracking_confidence expects a float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        logger.warning("Function encountered a ValueError for input parameter min_tracking_confidence. "
                       "Message: parameter min_tracking_confidence must be a float in the range [0,1].")
        raise ValueError("Apply_noise: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Logging input parameters
    if isinstance(noise_method, str):
        mask_type_name = get_variable_name(mask_type, globals())
        logger.info(f"Input parameters: noise_method = {noise_method}, pixel_size = {pixel_size}, noise_prob = {noise_prob}, "
                    f"rand_seed = {rand_seed}, mean = {mean}, standard_dev = {standard_dev}, mask_type = {mask_type_name}.")
    else:
        noise_method_name = get_variable_name(noise_method, globals())
        mask_type_name = get_variable_name(mask_type, globals())
        logger.info(f"Input parameters: noise_method = {noise_method_name}, pixel_size = {pixel_size}, noise_prob = {noise_prob}, "
                    f"rand_seed = {rand_seed}, mean = {mean}, standard_dev = {standard_dev}, mask_type = {mask_type_name}.")

    logger.info(f"Mediapipe configurations: min_detection_confidence = {min_detection_confidence}, "
                f"min_tracking_confidence = {min_tracking_confidence}.")
            
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = get_directory_walk(input_dir, with_sub_dirs)
        
    logger.info(f"Function read in {len(files_to_process)} files from input directory {input_dir}.")
    
    # Creating named output directories for video output
    output_dir = create_output_dir(output_dir, "Noise_Added")
    
    for file in files_to_process:
            
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        dir_file_path = output_dir + f"\\{filename}_noise_added{extension}"

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
        
        if not static_image_mode:
            capture = get_video_capture(file)
            size = (int(capture.get(3)), int(capture.get(4)))

            result = get_video_writer(dir_file_path, size, codec=codec)

        # Main Processing loop for video files (will only iterate once over images)
        while True:
            frame = None
            if static_image_mode:
                frame = cv.imread(file)
                if frame is None:
                    logger.error("Function has encountered an error attempting to read in a file. "
                                 f"Message: failed to read in file {file}.")
                    debug_logger.error("Function has encountered an error attempting to call cv2.imread(file). "
                                       f"Message: failed to read in file {file}. The file may be corrupt or incorrectly encoded.")
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
                logger.error("Face mesh detection error, function exiting with status 1.")
                debug_logger.error("Function encountered an error attempting to call mediapipe.face_mesh.FaceMesh.process() on the current frame.")
                raise FaceNotFoundError()
            
            output_frame = frame.copy()
            if isinstance(noise_method, str):
                noise_method = str.lower(noise_method)

            match noise_method:
                case 'pixelate' | 18:
                    height, width = output_frame.shape[:2]
                    h = frame.shape[0]//pixel_size
                    w = frame.shape[1]//pixel_size

                    temp = cv.resize(frame, (w, h), None, 0, 0, cv.INTER_LINEAR)
                    output_frame = cv.resize(temp, (width, height), None, 0, 0, cv.INTER_NEAREST)

                    img_mask = mask_frame(frame, face_mesh, mask_type)
                    output_frame = np.where(img_mask != 255, output_frame, frame)

                case 'salt and pepper' | 19:
                    # Divide prob in 2 for "salt" and "pepper"
                    thresh = noise_prob
                    noise_prob = noise_prob/2
                    rng = None

                    if rand_seed is not None:
                        rng = np.random.default_rng(rand_seed)
                    else:
                        rng = np.random.default_rng()
                    
                    # Use numpy's random number generator to generate a random matrix in the shape of the frame
                    rdm = rng.random(output_frame.shape[:2])

                    # Create boolean masks 
                    pepper_mask = rdm < noise_prob
                    salt_mask = (rdm >= noise_prob) & (rdm < thresh)
                    
                    # Apply boolean masks
                    output_frame[pepper_mask] = [0,0,0]
                    output_frame[salt_mask] = [255,255,255]

                    img_mask = mask_frame(frame, face_mesh, mask_type)
                    output_frame = np.where(img_mask != 255, output_frame, frame)
                
                case 'gaussian' | 20:
                    var = standard_dev**2
                    rng = None

                    if rand_seed is not None:
                        rng = np.random.default_rng(rand_seed)
                    else:
                        rng = np.random.default_rng()

                    # scikit-image's random_noise function works with floating point images, need to convert our frame's type
                    output_frame = img_as_float64(output_frame)
                    output_frame = random_noise(image=output_frame, mode='gaussian', rng=rng, mean=mean, var=var)
                    output_frame = img_as_ubyte(output_frame)

                    img_mask = mask_frame(frame, face_mesh, mask_type)
                    output_frame = np.where(img_mask != 255, output_frame, frame)

                case _:
                    logger.warning("Function has encountered an unrecognized value for parameter noise_method during execution, "
                                   "exiting with status 1. Input parameter checks may not be functioning as expected.")
                    raise ValueError("Unrecognized value for parameter noise_method.")

            if not static_image_mode:
                result.write(output_frame)
            else:
                success = cv.imwrite(output_dir + "\\" + filename + "_noise_added" + extension, output_frame)
                if not success:
                    logger.error("Function has encountered an error attempting to call cv2.imwrite(), exiting with status 1.")
                    debug_logger.error("Function has encountered an error attempting to call cv2.imwrite() to directory "
                                       f"{dir_file_path}. Please ensure the output directory path is a valid path in your current working directory tree.")
                    raise FileWriteError()
                break
        
        if not static_image_mode:
            capture.release()
            result.release()
    
        logger.info(f"Function execution completed successfully. View outputted file(s) at {output_dir}.")