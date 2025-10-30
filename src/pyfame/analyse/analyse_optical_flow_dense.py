from pydantic import BaseModel, ValidationError, PositiveInt, NonNegativeInt, PositiveFloat, NonNegativeFloat
from pyfame.landmark.facial_landmarks import *
from pyfame.file_access import get_video_capture
from pyfame.utilities.exceptions import *
from pyfame.utilities.constants import *
from pyfame.file_access.checks import *
import os
import cv2 as cv
import numpy as np
import pandas as pd
from skimage.util import *

class DenseFlowAnalysisParameters(BaseModel):
    pixel_neighborhood_size:PositiveInt = 5
    search_window_size:PositiveInt = 15
    max_pyramid_level:NonNegativeInt = 2
    pyramid_scale:PositiveFloat = 0.5
    max_iterations:PositiveInt = 10
    gaussian_deviation:NonNegativeFloat = 1.2
    frame_step:PositiveInt = 5

def analyse_optical_flow_dense(file_paths:pd.DataFrame, frame_step:int = 5) -> dict[str, pd.DataFrame]:
    '''Takes an input video file, and computes the dense optical flow, outputting the visualised optical flow to output_dir.
    Dense optical flow uses Farneback's algorithm to track every point within a frame.

    Parameters
    ----------

    file_paths: pandas.DataFrame
        An Nx2 dataframe of absolute and relative file paths, returned by the make_paths() function.
    
    frame_step: int
        The number of frames between successive optical flow calculations. The flow values will be more consistent 
        and robust as you increase this parameter. 
    
    Returns
    -------

    dict[str, pandas.DataFrame]

    Raises
    ------

    ValidationError:
        Thrown by the pydantic model when invalid parameters are passed to the method.
    
    FileReadError:
        When the working directory path; or any of its required sub-paths cannot be located. 

    UnrecognizedExtensionError
        If an image file is passed; Farneback's dense flow requires video files.
    '''

    # Validate and assign input parameters
    try:
        input_parameters = DenseFlowAnalysisParameters(frame_step=frame_step)
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {analyse_optical_flow_dense.__name__}: {e}")

    pyramid_scale = input_parameters.pyramid_scale
    max_pyramid_level = input_parameters.max_pyramid_level
    search_window_size = input_parameters.search_window_size
    max_iterations = input_parameters.max_iterations
    pixel_neighborhood_size = input_parameters.pixel_neighborhood_size
    gaussian_deviation = input_parameters.gaussian_deviation

    # Extracting the i/o paths from the file_paths dataframe
    absolute_paths = file_paths["Absolute Path"]

    norm_path = os.path.normpath(absolute_paths.iloc[0])
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
        counter = 0
        old_gray = None

        # Main Processing loop
        while True:
            counter += 1

            # Cache the last frame before reassigning the variable
            if counter > 1:
                old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            success, frame = capture.read()
            if not success:
                break    
            
            if counter % frame_step != 0:
                continue
            
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            timestamp = capture.get(cv.CAP_PROP_POS_MSEC)

            # Calculate dense optical flow
            flow = cv.calcOpticalFlowFarneback(old_gray, gray_frame, None, pyramid_scale, max_pyramid_level, search_window_size, max_iterations, pixel_neighborhood_size, gaussian_deviation, 0)

            # Get vector magnitudes and angles
            magnitudes, angles = cv.cartToPolar(flow[...,0],flow[...,1])

            # Get magnitude/angle means and distribution
            mean_mag = np.mean(magnitudes)
            std_mag = np.std(magnitudes)
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)

            # Dataframes are immutable, so we need to store as lists during execution
            timestamps.append(timestamp/1000)
            mean_magnitudes.append(mean_mag)
            std_magnitudes.append(std_mag)
            mean_angles.append(mean_angle)
            std_angles.append(std_angle)

            old_gray = gray_frame.copy()

        capture.release()

        output_df = pd.DataFrame({
            "timestamp":timestamps,
            "mean magnitude":mean_magnitudes,
            "deviation magnitude":std_magnitudes,
            "mean angle":mean_angles,
            "deviation angle":std_angles
        })
        outputs.update({f"{filename}":output_df})
    
    return outputs