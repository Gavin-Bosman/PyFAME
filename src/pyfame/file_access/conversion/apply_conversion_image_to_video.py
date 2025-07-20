from pyfame.utilities.exceptions import *
from pyfame.utilities.checks import *
from pyfame.utilities.constants import *
from pyfame.file_access import create_output_directory, get_video_writer, get_directory_walk
import numpy as np
import cv2 as cv
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def standardise_image_dimensions(input_directory:str, method:int|str = STANDARDISE_DIMS_CROP, pad_colour:tuple[int] = (255,255,255),
                       with_sub_dirs:bool = False) -> None:
    """ Takes in an input directory of static images, then scans through the directory to compute the maximal and minimal 
    image dimensions. Depending on the equalization method provided, each image in the directory will be either padded to the maximal 
    dimensions, or cropped to the minimal dimensions.

    Parameters:
    -----------

    input_directory: str
        A path string to a directory containing static images to be equalized.

    method: int
        An integer flag specifying the equalization method to use; one of STANDARDIZE_DIMS_CROP or STANDARDIZE_DIMS_PAD.
    
    pad_colour: tuple of int
        A BGR color code specifying the fill color of the padded region added to each image when using STANDARDIZE_DIMS_PAD.
    
    with_sub_dirs: bool
        A boolean flag indicating if the input directory contains nested subdirectories.

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
    
    Returns
    -------

    None

    """
    
    logger.info("Now entering function Equate_image_sizes.")
    check_type(input_directory, [str])
    check_valid_path(input_directory)

    check_type(method, [int,str])
    if isinstance(method, str):
        method = str.lower(method)
    check_value(method, [STANDARDISE_DIMS_CROP, STANDARDISE_DIMS_PAD, "crop", "pad"])

    check_type(pad_colour, [tuple])
    check_type(pad_colour, [int], iterable=True)

    check_type(with_sub_dirs, [bool])
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = get_directory_walk(input_directory, with_sub_dirs)

    min_size = None
    max_size = None

    for file in files_to_process:
        frame = cv.imread(file)
        if frame is None:
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
    match method:
        case 41 | "crop":
            for file in files_to_process:
                img = cv.imread(file, cv.IMREAD_COLOR_BGR)
                if img is None:
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
                
        case 42 | "pad":
            for file in files_to_process:
                img = cv.imread(file, cv.IMREAD_COLOR_BGR)
                if img is None:
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
                    bordered = cv.copyMakeBorder(img, margin_t, margin_b, margin_t, margin_b, cv.BORDER_CONSTANT, value=pad_colour)
                    bordered = bordered.astype(np.uint8)
                    h = h + margin_t + margin_b
                    img = bordered[:, margin_t:w + margin_t]
                if diff_w > 0:
                    bordered = cv.copyMakeBorder(img, margin_l, margin_r, margin_l, margin_r, cv.BORDER_CONSTANT, value=pad_colour)
                    bordered = bordered.astype(np.uint8)
                    img = bordered[margin_l:h + margin_l, :]

                # writing output image
                success = cv.imwrite(file, img)
                if not success:
                    raise FileWriteError()

def apply_conversion_image_to_video(input_directory:str, output_directory:str, output_filename:str, frame_rate:int = 30, repeat_duration_msec:int = 1000, 
                                    blend_transition:bool = True, blended_frames_proportion:float = 0.2, standardize_dimensions:bool = False, 
                                    standardize_method:int|str = STANDARDISE_DIMS_CROP, pad_colour:tuple[int] = (255,255,255), with_sub_dirs:bool = False) -> None:
    """ Takes a series of static images contained in input_dir, and converts them into a video sequence by repeating
    and interpolating frames. Output "movie" files will be written to output_dir. The output video file will have the images
    written in the order they appear within the input directory.

    Parameters:
    -----------

    input_directory: str
        The path string to the directory containing the images to be moviefied.

    output_directory: str
        The path string to the directory where the output video will be written too.

    output_filename: str
        A string specifying the name of the output video file. 

    frame_rate: int
        The frames per second of the output video file, passed as a parameter to cv2.VideoWriter().
    
    repeat_duration_msec: int
        The duration in milliseconds for each image to be repeated. i.e. with a repeat_duration of 1000, and an fps of 30,
        each image would be repeated 30 times (without blending).
    
    blend_transition: bool
        A boolean flag specifying if each image should blend into the next at the end of the repeat window.
    
    blended_frames_proportion: float
        A float in the range [0,1] specifying how much of an images repeat window should be used for the blending transition.
    
    standardize_dimensions: bool
        A boolean flag specifying if the input image sizes need to be equalized.

    standardize_method: int
        An integer flag, one of EQUATE_IMAGES_CROP or EQUATE_IMAGES_PAD. Specifies the equalization_method to use
        with equate_image_sizes().
    
    pad_color: tuple of int
        A BGR color code specifying the color of the padded image borders added if equalization_method is set to EQUATE_IMAGES_PAD.
    
    with_sub_dirs: bool
        A boolean flag specifying if the input directory contains nested sub directories.
    
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
    
    Returns:
    --------

    None
    
    """

    check_type(input_directory, [str])
    check_valid_path(input_directory)

    check_type(output_directory, [str])
    check_valid_path(output_directory)
    check_is_dir(output_directory)

    check_type(output_filename, [str])

    check_type(frame_rate, [int])
    check_value(frame_rate, min=1, max=120) 

    check_type(repeat_duration_msec, [int])
    check_value(repeat_duration_msec, min=100)

    check_type(blend_transition, [bool])
    
    check_type(blended_frames_proportion, [float])
    check_value(blended_frames_proportion, min=0.0, max=1.0)

    check_type(standardize_dimensions, [bool])

    check_type(standardize_method, [int,str])
    check_value(standardize_method, [STANDARDISE_DIMS_CROP, STANDARDISE_DIMS_PAD, "crop", "pad"])

    check_type(pad_colour, [tuple])
    check_type(pad_colour, [int], iterable=True)

    check_type(with_sub_dirs, [bool])
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = get_directory_walk(input_directory, with_sub_dirs)
    
    output_directory = create_output_directory(output_directory, "Image_To_Video")
    
    image_list = []
    im_size = None

    for file in files_to_process:
        frame = cv.imread(file)
        if frame is None:
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
                if standardize_dimensions:
                    standardise_image_dimensions(input_directory=input_directory, method=standardize_method, pad_colour=pad_colour)
                else:
                    raise ImageShapeError()
    
    # Instantiating our videowriter
    writer = get_video_writer(output_directory + f"\\{output_filename}.mp4", frame_size=im_size)
    
    num_repeats = int(np.floor((repeat_duration_msec/1000)*frame_rate))
    blend_window = round(num_repeats * blended_frames_proportion)
    blend_inc = 1/blend_window
    
    for i in range(len(image_list)):
        cur_inc = 0.0
        for j in range(num_repeats):
            if blend_transition == True:
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