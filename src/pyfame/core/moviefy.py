from pyfame.utils.exceptions import *
from pyfame.utils.predefined_constants import *
from pyfame.utils.utils import get_variable_name
import numpy as np
import cv2 as cv
import os
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def equate_image_sizes(input_dir:str, method:int = EQUATE_IMAGES_CROP, pad_color:tuple[int] = (255,255,255),
                          with_sub_dirs:bool = False) -> None:
    """ Takes in an input directory of static images, then scans through the directory to compute the maximal and minimal 
    image dimensions. Depending on the equalization method provided, each image in the directory will be either padded to the maximal 
    dimensions, or cropped to the minimal dimensions.

    Parameters:
    -----------

    input_dir: str
        A path string to a directory containing static images to be equalized.

    method: int
        An integer flag specifying the equalization method to use; one of EQUATE_IMAGES_PAD, EQUATE_IMAGES_CROP.
    
    pad_color: tuple of int
        A BGR color code specifying the fill color of the padded region added to each image when using EQUATE_IMAGES_PAD.
    
    with_sub_dirs: bool
        A boolean flag indicating if the input directory contains nested subdirectories
    
    Returns:
    --------
    
    None

    Raises:
    -------

    TypeError:
        Given invalid parameter typings.
    ValueError:
        Given invalid parameter values.
    OSError: 
        Given invalid path strings to input_dir.
    FileWriteError:
        On error catches thrown by cv2.imwrite or cv2.VideoWriter.
    FileReadError:
        On error catches thrown by cv2.imread.

    """
    single_file = False
    logger.info("Now entering function Equate_image_sizes.")

    # Performing input parameter checks
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: parameter input_dir must be of type str.")
        raise TypeError("Equate_image_sizes: parameter input_dir must be of type str.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path, or the directory does not exist.")
        raise OSError("Equate_image_sizes: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        single_file = True

    if not isinstance(method, int):
        logger.warning("Function encountered a TypeError for input parameter method. "
                       "Message: parameter normalization_method must be an int.")
        raise TypeError("Equate_image_sizes: parameter method must be an int.")
    elif method not in [EQUATE_IMAGES_CROP, EQUATE_IMAGES_PAD]:
        logger.warning("Function encountered a ValueError for input parameter method. "
                       "Message: unrecognized value for parameter method.")
        raise ValueError("Equate_image_sizes: unrecognized value for parameter method.")
    
    if not isinstance(pad_color, tuple):
        logger.warning("Function encountered a TypeError for input parameter pad_color. "
                       "Message: parameter pad_color must be a tuple of int.")
        raise TypeError("Equate_image_sizes: parameter pad_color must be a tuple of int.")
    for val in pad_color:
        if not isinstance(val, int):
            logger.warning("Function encountered a TypeError for input parameter pad_color. "
                       "Message: parameter pad_color must be a tuple of int.")
            raise TypeError("Equate_image_sizes: parameter pad_color must be a tuple of int.")
        if val < 0 or val > 255:
            logger.warning("Function encountered a ValueError for input parameter pad_color. "
                           "Message: parameter pad_color must contain integers in the range 0-255.")
            raise ValueError("Equate_image_sizes: BGR values may only fall in the range 0-255.")
    
    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: parameter with_sub_dirs must be of type bool.")
        raise TypeError("Equate_image_sizes: parameter with_sub_dirs must be of type bool.")
    
    # Logging input parameters
    eq_meth_name = get_variable_name(method, globals())
    logger.info(f"Input Parameters: input_dir = {input_dir}, method = {eq_meth_name}, pad_color = {pad_color}, with_sub_dirs = {with_sub_dirs}.")

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

    min_size = None
    max_size = None

    for file in files_to_process:
        frame = cv.imread(file)
        if frame is None:
            debug_logger.error("Function has encountered a FileReadError. "
                               "Message: Function encountered an error attempting to call cv2.imread() over"
                               f"file {file}.")
            raise FileReadError()
        
        if min_size is None:
            h, w, ch = frame.shape
            min_size = [h, w, ch]
        
        if max_size is None:
            h, w, ch = frame.shape
            max_size = [h, w, ch]
        
        cur_size = [frame.shape[0], frame.shape[1], frame.shape[2]]
        if cur_size[0] < min_size[0]:
            min_size[0] = cur_size[0]
        if cur_size[1] < min_size[1]:
            min_size[1] = cur_size[1]
        if cur_size[0] > max_size[0]: 
            max_size[0] = cur_size[0]
        if cur_size[1] > max_size[1]:
            max_size[1] = cur_size[1]
    # resizing the images
    if method == EQUATE_IMAGES_CROP:
        for file in files_to_process:
            img = cv.imread(file, cv.IMREAD_COLOR_BGR)
            if img is None:
                debug_logger.error("Function has encountered a FileReadError. "
                               "Message: Function encountered an error attempting to call cv2.imread() over"
                               f"file {file}.")
                raise FileReadError()
            
            h, w, ch = img.shape
            h2, w2, ch2 = min_size
            diff_h = abs(h-h2)
            diff_w = abs(w-w2)

            crop_margin_h = diff_h/2
            crop_margin_w = diff_w/2
            margin_l = 0
            margin_r = 0
            margin_t = 0
            margin_b = 0

            if isinstance(crop_margin_h, float):
                margin_t = int(np.ceil(crop_margin_h))
                margin_b = int(np.floor(crop_margin_h))
            else:
                margin_t = crop_margin_h
                margin_b = crop_margin_h

            if isinstance(crop_margin_w, float):
                margin_l = int(np.ceil(crop_margin_w))
                margin_r = int(np.floor(crop_margin_w))
            else:
                margin_l = crop_margin_w
                margin_r = crop_margin_w

            if diff_h > 0:
                img = img[margin_t:(h-margin_b), :]
            if diff_w > 0:    
                img = img[:, margin_l:(w-margin_r)]
                
            # writing output image
            success = cv.imwrite(file, img)
            if not success:
                raise FileWriteError()
    else:
        for file in files_to_process:
            img = cv.imread(file, cv.IMREAD_COLOR_BGR)
            if img is None:
                debug_logger.error("Function has encountered a FileReadError. "
                               "Message: Function encountered an error attempting to call cv2.imread() over"
                               f"file {file}.")
                raise FileReadError()
            
            h, w, ch = img.shape
            h2, w2, ch2 = max_size
            diff_h = abs(h-h2)
            diff_w = abs(w-w2)
            
            pad_margin_h = diff_h/2
            pad_margin_w = diff_w/2
            margin_l = 0
            margin_r = 0
            margin_t = 0
            margin_b = 0

            if isinstance(pad_margin_h, float):
                margin_t = int(np.ceil(pad_margin_h))
                margin_b = int(np.floor(pad_margin_h))
            else:
                margin_t = pad_margin_h
                margin_b = pad_margin_h

            if isinstance(pad_margin_w, float):
                margin_l = int(np.ceil(pad_margin_w))
                margin_r = int(np.floor(pad_margin_w))
            else:
                margin_l = pad_margin_w
                margin_r = pad_margin_w

            if diff_h > 0:
                bordered = cv.copyMakeBorder(img, margin_t, margin_b, margin_t, margin_b, cv.BORDER_CONSTANT, value=pad_color)
                bordered = bordered.astype(np.uint8)
                h = h + margin_t + margin_b
                img = bordered[:, margin_t:w + margin_t]
            if diff_w > 0:
                bordered = cv.copyMakeBorder(img, margin_l, margin_r, margin_l, margin_r, cv.BORDER_CONSTANT, value=pad_color)
                bordered = bordered.astype(np.uint8)
                img = bordered[margin_l:h + margin_l, :]

            # writing output image
            success = cv.imwrite(file, img)
            if not success:
                debug_logger.error("Function encountered a FileWriteError. "
                           "Message: function encountered an error attempting to call cv2.imwrite over "
                           f"file {file}.")
                raise FileWriteError()
    logger.info("Function execution completed successfully.")


def moviefy_images(input_dir:str, output_dir:str, output_filename:str, fps:int = 30, repeat_duration:int = 1000, 
    blend_images:bool = True, blended_frames_prop:float = 0.2, equate_sizes:bool = False, 
    equalization_method:int = EQUATE_IMAGES_CROP, pad_color:tuple[int] = (255,255,255), with_sub_dirs:bool = False) -> None:
    """ Takes a series of static images contained in input_dir, and converts them into a video sequence by repeating
    and interpolating frames. Output "movie" files will be written to output_dir. The output video file will have the images
    written in the order they appear within the input directory.

    Parameters:
    -----------

    input_dir: str
        The path string to the directory containing the images to be moviefied.

    output_dir: str
        The path string to the directory where the output video will be written too.

    output_filename: str
        A string specifying the name of the output video file. 

    fps: int
        The frames per second of the output video file, passed as a parameter to cv2.VideoWriter().
    
    repeat_duration: int
        The duration in milliseconds for each image to be repeated. i.e. with a repeat_duration of 1000, and an fps of 30,
        each image would be repeated 30 times (without blending).
    
    blend_images: bool
        A boolean flag specifying if each image should blend into the next at the end of the repeat window.
    
    blended_frames_prop: float
        A float in the range [0,1] specifying how much of an images repeat window should be used for the blending transition.
    
    equate_sizes: bool
        A boolean flag specifying if the input image sizes need to be equalized.

    equalization_method: int
        An integer flag, one of EQUATE_IMAGES_CROP or EQUATE_IMAGES_PAD. Specifies the equalization_method to use
        with equate_image_sizes().
    
    pad_color: tuple of int
        A BGR color code specifying the color of the padded image borders added if equalization_method is set to EQUATE_IMAGES_PAD.
    
    with_sub_dirs: bool
        A boolean flag specifying if the input directory contains nested sub directories.
    
    Returns:
    --------

    None
    
    Raises:
    -------

    TypeError:
        Given invalid parameter typings.
    ValueError:
        Given invalid parameter values.
    OSError: 
        Given invalid path strings to input/output_dir.
    ImageShapeError:
        Given mismatching input image shapes.
    FileWriteError:
        On error catches thrown by cv2.imwrite or cv2.VideoWriter.
    FileReadError:
        On error catches thrown by cv2.imread.
    
    """
    single_file = False
    logger.info("Now entering function moviefy_images.")

    # Performing input parameter checks
    if not isinstance(input_dir, str):
        logger.warning("Function encountered a TypeError for input parameter input_dir. "
                       "Message: parameter input_dir must be of type str.")
        raise TypeError("Moviefy_images: parameter input_dir must be of type str.")
    elif not os.path.exists(input_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path, or the directory does not exist.")
        raise OSError("Moviefy_images: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        single_file = True
    
    if not isinstance(output_dir, str):
        logger.warning("Function encountered a TypeError for input parameter output_dir. "
                       "Message: parameter output_dir must be of type str.")
        raise TypeError("Moviefy_images: parameter output_dir must be of type str.")
    elif not os.path.exists(output_dir):
        logger.warning("Function encountered an OSError for input parameter input_dir. "
                       "Message: input_dir is not a valid path, or the directory does not exist.")
        raise OSError("Moviefy_images: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_dir):
        logger.warning("Function encountered a ValueError for input parameter output_dir. "
                       "Message: output_dir must be a valid path to a directory.")
        raise ValueError("Moviefy_images: output_dir must be a valid path to a directory.")
    
    if not isinstance(output_filename, str):
        logger.warning("Function encountered a TypeError for input parameter output_filename. "
                       "Message: parameter output_filename must be a str.")
        raise TypeError("Moviefy_images: parameter output_filename must be of type str.")
    
    if not isinstance(fps, int):
        logger.warning("Function encountered a TypeError for input parameter fps. "
                       "Message: parameter fps must be an int.")
        raise TypeError("Moviefy_images: parameter fps must be an int.")
    elif fps < 0 or fps > 120:
        logger.warning("Function encountered a ValueError for input parameter fps. "
                       "Message: fps must lie in the range [1,120]. ")
        raise ValueError("Moviefy_images: parameter fps must lie between 1-120. ")
    
    if not isinstance(repeat_duration, int):
        logger.warning("Function encountered a TypeError for input parameter repeat_duration. "
                       "Message: parameter repeat_duration must be an int.")
        raise TypeError("Moviefy_images: parameter repeat_duration must be an int.")
    elif repeat_duration < 100:
        logger.warning("Function encountered a ValueError for input parameter repeat_duration. "
                       "Message: parameter repeat_duration must be an int > 100.")
        raise ValueError("Moviefy_images: parameter repeat_duration cannot be less than 100 msec.")
    
    if not isinstance(blend_images, bool):
        logger.warning("Function encountered a TypeError for input parameter blend_images. "
                       "Message: parameter blend_images must be of type bool.")
        raise TypeError("Moviefy_images: parameter blend_images must be of type bool.")
    
    if not isinstance(blended_frames_prop, float):
        logger.warning("Function encountered a TypeError for input parameter blended_frames_prop. "
                       "Message: parameter blended_frames_prop must be of type float.")
        raise TypeError("Moviefy_images: parameter blended_frames_prop must be of type float.")
    elif blended_frames_prop < 0 or blended_frames_prop > 1:
        logger.warning("Function encountered a ValueError for input parameter blended_frames_prop. "
                       "Message: parameter blended_frames_prop must be a float in the range [0,1].")
        raise ValueError("Moviefy_images: parameter blended_frames_prop must be a float in the range [0,1].")
    
    if not isinstance(equate_sizes, bool):
        logger.warning("Function encountered a TypeError for input parameter normalize. "
                       "Message: parameter normalize must be of type bool.")
        raise TypeError("Moviefy_images: parameter normalize must be of type bool.")
    
    if not isinstance(equalization_method, int):
        logger.warning("Function encountered a TypeError for input parameter normalization_method. "
                       "Message: parameter normalization_method must be an int.")
        raise TypeError("Moviefy_images: parameter normalization_method must be an int.")
    elif equalization_method not in [EQUATE_IMAGES_CROP, EQUATE_IMAGES_PAD]:
        logger.warning("Function encountered a ValueError for input parameter normalization_method. "
                       "Message: unrecognized value for parameter normalization_method.")
        raise ValueError("Moviefy_images: unrecognized value for parameter normalization_method.")
    
    if not isinstance(with_sub_dirs, bool):
        logger.warning("Function encountered a TypeError for input parameter with_sub_dirs. "
                       "Message: parameter with_sub_dirs must be of type bool.")
        raise TypeError("Moviefy_images: parameter with_sub_dirs must be of type bool.")
    
    # Logging input parameters
    norm_meth_name = get_variable_name(equalization_method, globals())
    logger.info(f"Input Parameters: input_dir = {input_dir}, output_dir = {output_dir}, output_filename = {output_filename}, "
                f"fps = {fps}, repeat_duration = {repeat_duration}, blend_images = {blend_images}, blended_frames_prop = {blended_frames_prop}, "
                f"normalize = {equate_sizes}, normalization_method = {norm_meth_name}, with_sub_dirs = {with_sub_dirs}.")

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
    if not os.path.isdir(output_dir + "\\Moviefied"):
        os.mkdir(output_dir + "\\Moviefied")
        output_dir = output_dir + "\\Moviefied"
        logger.info(f"Function created new output directory {output_dir}.")
    else:
        output_dir = output_dir + "\\Moviefied"
    
    image_list = []
    im_size = None

    for file in files_to_process:
        frame = cv.imread(file)
        if frame is None:
            debug_logger.error("Function has encountered a FileReadError. "
                               "Message: Function encountered an error attempting to call cv2.imread() over"
                               f"file {file}.")
            raise FileReadError()
        
        image_list.append(frame)

        if im_size is None:
            # VideoWriter expects (width, height)
            im_size = (frame.shape[1], frame.shape[0])
    
    # Before proceeding, need to check that all input images have the same size
    im_shape = None
    for image in image_list:
        if im_shape is None:
            im_shape = image.shape
        else:
            if im_shape != image.shape:
                if equate_sizes:
                    equate_image_sizes(input_dir=input_dir, method=equalization_method, pad_color=pad_color)
                else:
                    debug_logger.error("Function encountered an ImageShapeError. "
                                       "Message: mismatching input image shapes cannot be processed. Please see "
                                       "Equate_image_sizes() or set normalize=True.")
                    raise ImageShapeError()
    
    # Instantiating our videowriter
    writer = cv.VideoWriter(output_dir + f"\\{output_filename}.mp4", cv.VideoWriter.fourcc(*"mp4v"), fps, im_size)

    if not writer.isOpened():
        debug_logger.error("Function encountered a FileWriteError. ")
                           
        raise FileWriteError()
    
    num_repeats = int(np.floor((repeat_duration/1000)*fps))
    blend_window = round(num_repeats * blended_frames_prop)
    blend_inc = 1/blend_window
    
    for i in range(len(image_list)):
        cur_inc = 0.0
        for j in range(num_repeats):
            if blend_images == True:
                if j > (num_repeats-blend_window) and (i < (len(image_list) - 1)):
                    # begin blending images
                    cur_inc += blend_inc
                    alpha = cur_inc
                    beta = (1-alpha)
                    blended_img = cv.addWeighted(image_list[i], beta, image_list[i+1], alpha, 0.0)
                    blended_img.astype(np.uint8)
                    writer.write(blended_img)
                else:
                    writer.write(image_list[i])
            else:
                writer.write(image_list[i])
    
    writer.release()
    logger.info(f"Function execution has completed successfully.")