from pyfame.utilities.exceptions import *
from pyfame.utilities.checks import *
from pyfame.utilities.constants import *
from pyfame.file_access import create_output_directory, get_video_writer, get_directory_walk
import numpy as np
import cv2 as cv
import pandas as pd

def standardise_image_dimensions(input_directory:str, method:int|str = STANDARDISE_DIMS_CROP, pad_colour:tuple[int,int,int] = (255,255,255)) -> None:
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
    
    check_type(input_directory, [str])
    check_valid_path(input_directory)

    check_type(method, [int,str])
    if isinstance(method, str):
        method = str.lower(method)
    check_value(method, [STANDARDISE_DIMS_CROP, STANDARDISE_DIMS_PAD, "crop", "pad"])

    check_type(pad_colour, [tuple])
    check_type(pad_colour, [int], iterable=True)
    
    # Creating a list of file path strings to iterate through when processing
    files_df = get_directory_walk(input_directory)
    files_to_process = files_df["Absolute Path"]

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

def apply_conversion_image_to_video(file_paths:pd.DataFrame, output_filename:str, frame_rate:int = 30, repeat_duration_msec:int = 1000, 
                                    blend_transition:bool = True, blended_frames_proportion:float = 0.2) -> None:
    """ Takes a series of static images contained in input_dir, and converts them into a video sequence by repeating
    and interpolating frames. Output "movie" files will be written to output_dir. The output video file will have the images
    written in the order they appear within the input directory.

    Parameters:
    -----------

    file_paths: DataFrame
        A 2-column dataframe consisting of absolute and relative file paths.

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

    check_type(output_filename, [str])

    check_type(frame_rate, [int])
    check_value(frame_rate, min=1, max=120) 

    check_type(repeat_duration_msec, [int])
    check_value(repeat_duration_msec, min=100)

    check_type(blend_transition, [bool])
    
    check_type(blended_frames_proportion, [float])
    check_value(blended_frames_proportion, min=0.0, max=1.0)

    # Extracting the i/o paths from the file_paths dataframe
    absolute_paths = file_paths["Absolute Path"]

    norm_path = os.path.normpath(absolute_paths[0])
    norm_cwd = os.path.normpath(os.getcwd())
    rel_dir_path, *_ = os.path.split(os.path.relpath(norm_path, norm_cwd))
    parts = rel_dir_path.split(os.sep)
    root_directory = None

    # extracting the root directory name to use in fileIO
    if parts is not None:
        root_directory = parts[0]
    
    if root_directory is None:
        root_directory = "data"
    
    # Ensure the correct structure has been set up 
    test_path = os.path.join(norm_cwd, root_directory)

    if not os.path.isdir(test_path):
        raise FileReadError(message=f"Unable to locate the input {root_directory} directory. Please call make_output_paths() to set up the correct directory structure.")
    if not os.path.isdir(os.path.join(test_path, "raw")):
        raise FileReadError(message=f"Unable to locate the 'raw' subdirectory under root directory '{root_directory}'. Please call make_output_paths() to set up the correct directory structure.")
    if not os.path.isdir(os.path.join(test_path, "processed")):
        raise FileReadError(message=f"Unable to locate the 'processed' subdirectory under root directory '{root_directory}'. Please call make_output_paths() to set up the correct directory structure.")
    
    # Creating a unique subdirectory for image->video conversion outputs
    output_directory = os.path.join(test_path, "processed")
    output_directory = create_output_directory(output_directory, "image_to_video")
    
    image_list = []
    im_size = None

    for file in absolute_paths:
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
                raise ImageShapeError(message="Image shapes provided do not match. Please call standardise_image_dimensions() over the directory containing the input images.")
    
    # Instantiating our videowriter
    writer = get_video_writer(file_path=os.path.join(output_directory, f"{output_filename}.mp4"), frame_size=im_size)
    
    num_repeats = int(np.floor((repeat_duration_msec/1000)*frame_rate))
    blend_window = round(num_repeats * blended_frames_proportion)
    blend_inc = 1/blend_window
    
    # Iterate over the list of frames
    for i in range(len(image_list)):
        cur_inc = 0.0
        # Each frame is repeated for a user-specified time duration
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