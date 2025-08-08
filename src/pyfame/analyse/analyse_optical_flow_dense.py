from pyfame.mesh.mesh_landmarks import *
from pyfame.file_access import get_video_capture
from pyfame.utilities.exceptions import *
from pyfame.utilities.constants import *
from pyfame.utilities.checks import *
import os
import cv2 as cv
import numpy as np
import pandas as pd
from skimage.util import *

def analyse_optical_flow_dense(file_paths:pd.DataFrame, block_size:int = 5, search_window_size:int = 15, max_pyramid_level:int = 2, 
                               pyramid_scale:float = 0.5, max_iterations:int = 10, gaussian_deviation:float = 1.2, 
                               with_sub_dirs:bool = False, output_sample_frequency_msec:int = 1000) -> None:
    '''Takes an input video file, and computes the dense optical flow, outputting the visualised optical flow to output_dir.
    Dense optical flow uses Farneback's algorithm to track every point within a frame.

    Parameters
    ----------

    input_directory: str
        A path string to a directory containing the video files to be processed.

    output_directory: str
        A path string to a directory where outputted csv files will be written to.
    
    block_size: int
        The size of the pixel neighborhood used in Farneback's dense optical flow algorithm.
    
    search_window_size: tuple of int
        The size of the search window (in pixels) used at each pyramid level in Lucas-Kanade sparse optical flow.

    max_pyramid_lvl: int
        The maximum number of pyramid levels used in Lucas Kanade sparse optical flow. As you increase this parameter larger motions can be 
        detected but consequently computation time increases.
    
    pyramid_scale: float
        A float in the range [0,1] representing the downscale of the image at each pyramid level in Farneback's dense optical flow algorithm.
        For example, with a pyr_scale of 0.5, at each pyramid level the image will be half the size of the previous image.
    
    max_iterations: int
        The maximum number of iterations (over each frame) the optical flow algorithm will make before terminating.

    gaussian_deviation: float
        A floating point value representing the standard deviation of the Gaussian distribution used in the polynomial expansion of Farneback's
        dense optical flow algorithm. Typically with block_sizes of 5 or 7, a gaussian_deviation of 1.2 or 1.5 are used respectively.

    with_sub_dirs: bool
        Indicates whether the input directory contains subfolders.

    output_sample_frequency_msec: int
        The time delay in milliseconds between successive csv write calls. Increase this value to speed up computation time, and decrease 
        the value to increase the number of optical flow vector samples written to the output csv file.
    
    Returns
    -------

    None
    '''

    # Performing parameter checks
    check_type(block_size, [int])

    check_type(search_window_size, [int])

    check_type(max_pyramid_level, [int])
    check_value(max_pyramid_level, min=1, max=5)

    check_type(max_iterations, [int])
    check_value(max_iterations, min=0)
    
    check_type(with_sub_dirs, [bool])

    check_type(output_sample_frequency_msec, [int])
    check_value(output_sample_frequency_msec, min=50, max=2000)

    # Extracting the i/o paths from the file_paths dataframe
    absolute_paths = file_paths["Absolute Path"]

    norm_path = os.path.normpath(absolute_paths[0])
    norm_cwd = os.path.normpath(os.getcwd())
    rel_dir_path, *_ = os.path.split(os.path.relpath(norm_path, norm_cwd))
    parts = rel_dir_path.split(os.sep)
    root_directory = None

    if parts is not None:
        root_directory = parts[0]
    
    if root_directory is None:
        root_directory = "data"
    
    test_path = os.path.join(norm_cwd, root_directory)

    if not os.path.isdir(test_path):
        raise FileReadError(message=f"Unable to locate the input {root_directory} directory. Please call make_output_paths() to set up the correct directory structure.")
    if not os.path.isdir(os.path.join(test_path, "raw")):
        raise FileReadError(message=f"Unable to locate the 'raw' subdirectory under root directory '{root_directory}'. Please call make_output_paths() to set up the correct directory structure.")
    if not os.path.isdir(os.path.join(test_path, "processed")):
        raise FileReadError(message=f"Unable to locate the 'processed' subdirectory under root directory '{root_directory}'. Please call make_output_paths() to set up the correct directory structure.")

    # Create the outputs dict outside of the main loop so it maintains a larger scope
    outputs = {}

    for file in absolute_paths:

        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        capture = None

        # Using the file extension to sniff video codec or image container for images
        if extension not in [".mp4", ".mov"]:
            print(f"Skipping unparseable file {os.path.basename(file)}.")
            continue
        
        # Instantiating video read/writers
        capture = get_video_capture(file)

        # creating lists to store output data
        timestamps = []
        mean_magnitudes = []
        std_magnitudes = []
        mean_angles = []
        std_angles = []

        # Defining persistent loop params
        counter = 1
        old_gray = None
        rolling_time_win = output_sample_frequency_msec

        # Main Processing loop
        while True:
            success, frame = capture.read()
            if not success:
                break    

            old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                
            if counter > 1:
                gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                timestamp = capture.get(cv.CAP_PROP_POS_MSEC)

                # Calculate dense optical flow
                flow = cv.calcOpticalFlowFarneback(old_gray, gray_frame, None, pyramid_scale, max_pyramid_level, search_window_size, max_iterations, block_size, gaussian_deviation, 0)

                # Get vector magnitudes and angles
                magnitudes, angles = cv.cartToPolar(flow[...,0],flow[...,1])

                if timestamp > rolling_time_win:
                    mean_mag = np.mean(magnitudes)
                    std_mag = np.std(magnitudes)
                    mean_angle = np.mean(angles)
                    std_angle = np.std(angles)

                    # Dataframes are immutable, so we need to store as lists during execution
                    timestamps.append(timestamp//1000)
                    mean_magnitudes.append(mean_mag)
                    std_magnitudes.append(std_mag)
                    mean_angles.append(mean_angle)
                    std_angles.append(std_angle)

                    rolling_time_win += output_sample_frequency_msec

                old_gray = gray_frame.copy()
            
            counter += 1

        capture.release()

        output_df = pd.DataFrame({
            "Timestamp":timestamps,
            "Mean Magnitude":mean_magnitudes,
            "Deviation Magnitude":std_magnitudes,
            "Mean Angle":mean_angles,
            "Deviation Angle":std_angles
        })
        outputs.update({f"{filename}{extension}":output_df})
    
    return outputs