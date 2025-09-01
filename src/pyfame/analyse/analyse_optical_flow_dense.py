from pydantic import BaseModel, field_validator, ValidationInfo, ValidationError, PositiveInt, NonNegativeInt, PositiveFloat, NonNegativeFloat
from pyfame.mesh.mesh_landmarks import *
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
    output_sample_frequency:PositiveInt = 1000

    @field_validator("pyramid_scale")
    @classmethod
    def check_normal_range(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if not (0.0 < value <= 1.0):
            raise ValueError(f"Parameter {field_name} must lie in the normalised range 0.0-1.0.")
        
        return value

def analyse_optical_flow_dense(file_paths:pd.DataFrame, output_sample_frequency_msec:int = 1000) -> dict[str, pd.DataFrame]:
    '''Takes an input video file, and computes the dense optical flow, outputting the visualised optical flow to output_dir.
    Dense optical flow uses Farneback's algorithm to track every point within a frame.

    Parameters
    ----------

    file_paths: pandas.DataFrame
        An Nx2 dataframe of absolute and relative file paths, returned by the make_paths() function.
    
    output_sample_frequency_msec: int
        The time delay in milliseconds between successive csv write calls. Increase this value to speed up computation time, and decrease 
        the value to increase the number of optical flow vector samples written to the output csv file.
    
    Returns
    -------

    dict[str, pd.DataFrame]

    Raises
    ------

    ValueError
        On the passing of unrecognized input parameter values.
    UnrecognizedExtensionError
        If an image file is passed; Farneback's dense flow requires video files.
    '''

    # Validate and assign input parameters
    try:
        input_parameters = DenseFlowAnalysisParameters(output_sample_frequency=output_sample_frequency_msec)
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
                flow = cv.calcOpticalFlowFarneback(old_gray, gray_frame, None, pyramid_scale, max_pyramid_level, search_window_size, max_iterations, pixel_neighborhood_size, gaussian_deviation, 0)

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
        outputs.update({f"{filename}":output_df})
    
    return outputs