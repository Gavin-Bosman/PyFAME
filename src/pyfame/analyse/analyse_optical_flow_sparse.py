from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, NonNegativeInt, PositiveFloat, PositiveInt
from typing import Optional, List, Tuple
from pyfame.mesh import get_mesh, get_mesh_coordinates
from pyfame.mesh.mesh_landmarks import *
from pyfame.layer.manipulations.mask import mask_from_path
from pyfame.file_access import get_video_capture
from pyfame.utilities.exceptions import *
from pyfame.utilities.constants import *
import os
import cv2 as cv
import numpy as np
import pandas as pd
from skimage.util import *

class SparseFlowAnalysisParameters(BaseModel):
    landmarks_to_track:Optional[List[NonNegativeInt]]
    max_points:PositiveInt
    point_quality_threshold:PositiveFloat = 0.3
    min_point_distance:NonNegativeInt = 7
    pixel_neighborhood_size:Tuple[NonNegativeInt, NonNegativeInt] = (5,5)
    search_window_size:Tuple[NonNegativeInt, NonNegativeInt] = (15,15)
    max_pyramid_level:NonNegativeInt = 2
    max_iterations:PositiveInt = 10
    flow_accuracy_threshold:PositiveFloat
    output_sample_frequency:PositiveInt = 1000
    output_detail_level:str
    min_detection_confidence:PositiveFloat
    min_tracking_confidence:PositiveFloat

    @field_validator("point_quality_threshold", "flow_accuracy_threshold", "min_detection_confidence", "min_tracking_confidence")
    @classmethod
    def check_normal_range(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if not (0.0 < value <= 1.0):
            raise ValueError(f"Parameter {field_name} must lie in the normalised range 0.0-1.0.")
    
    @field_validator("output_detail_level")
    @classmethod
    def check_accepted_value(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if value not in {"summary", "full"}:
            raise ValueError(f"Unrecognized value for parameter {field_name}.")

def analyse_optical_flow_sparse(file_paths:pd.DataFrame, landmarks_to_track:list[int]|None = None, max_points:int = 20, 
                                flow_accuracy_threshold:float = 0.03, output_detail_level:str = "summary", output_sample_frequency:int = 1000,
                                min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> dict[str, pd.DataFrame]:
    
    """Takes each input video file provided within input_directory, and generates a sparse optical flow image, as well as a csv containing periodically
    sampled flow vector data. This function makes use of the Lucas-Kanadae optical flow algorithm, as well as the Shi-Tomasi good-corners algorithm to identify
    and track relevant points in the input video. Alternatively, specific facial landmarks to track can be passed in via landmarks_to_track.
    
    Parameters
    ----------
    
    file_paths: DataFrame
        A 2-column dataframe consisting of absolute and relative file paths.
    
    landmarks_to_track: list of int
        A list of mediapipe FaceMesh landmark id's, specifying relevant facial landmarks to track.
    
    max_points: int
        The maximum number of corners or "good points" for the Shi-Tomasi corners algorithm.

    flow_accuracy_threshold: float
        A termination criteria for Lucas-Kanadae optical flow; the algorithm will continue to iterate until this threshold is reached.

    output_detail_level: str
        Either "summary" specifying summary statisitics or "deep" specifying full descriptive output for each vector.
    
    output_sample_frequency: int
        The time duration (in msec) between successive samplings.
    
    min_detection_confidence: float
        A normalized float; an input parameter to the mediapipe FaceMesh solution.
    
    min_tracking_confidence: float
        A normalized float; an input parameter to the mediapipe FaceMesh solution.
    
    Returns
    -------

    None

    """
    
    # Validate and assign input parameters
    try:
        input_parameters = SparseFlowAnalysisParameters(
            landmarks_to_track=landmarks_to_track, 
            max_points=max_points,
            flow_accuracy_threshold=flow_accuracy_threshold,
            output_detail_level=output_detail_level,
            output_sample_frequency=output_sample_frequency,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {analyse_optical_flow_sparse.__name__}: {e}")
    
    output_sample_frequency_msec = input_parameters.output_sample_frequency
    search_window_size = input_parameters.search_window_size
    max_pyramid_level = input_parameters.max_pyramid_level
    max_iterations = input_parameters.max_iterations
    point_quality_threshold = input_parameters.point_quality_threshold
    min_point_distance = input_parameters.min_point_distance
    pixel_neighborhood_size = input_parameters.pixel_neighborhood_size
    
    # Defining the mediapipe facemesh task
    face_mesh = get_mesh(min_tracking_confidence, min_detection_confidence, False)
    
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
        magnitudes = []
        mean_magnitudes = []
        std_magnitudes = []
        angles = []
        mean_angles = []
        std_angles = []
        num_points = []
        full_stats = []
        
        # Defining persistent loop params
        counter = 0
        init_points = None
        old_gray = None
        rolling_time_win = output_sample_frequency_msec

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize  = search_window_size,
            maxLevel = max_pyramid_level,
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, max_iterations, flow_accuracy_threshold))

        # Main Processing loop
        while True:
            counter += 1
            success, frame = capture.read()
            if not success:
                break    
            
            # Get the landmark screen coordinates
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            landmark_screen_coords = get_mesh_coordinates(frame_rgb, face_mesh)
            
            # Create face oval image mask
            face_mask = mask_from_path(frame, FACE_OVAL_PATH, face_mesh)

            if counter == 1:
                # Get initial tracking points
                old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # If landmarks were provided 
                if landmarks_to_track is not None:
                    init_points = []
                    init_points = np.array([[lm.get('x'), lm.get('y')] for lm in landmark_screen_coords if lm.get('id') in landmarks_to_track], dtype=np.float32)
                    init_points = init_points.reshape(-1,1,2)
                else:
                    init_points = cv.goodFeaturesToTrack(old_gray, max_points, point_quality_threshold, min_point_distance, pixel_neighborhood_size, mask=face_mask)
                
            if counter > 1:
                gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                timestamp = capture.get(cv.CAP_PROP_POS_MSEC)

                # Calculate optical flow
                cur_points, st, err = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, init_points, None, **lk_params)

                # Select good points
                good_new_points = None
                good_old_points = None
                if cur_points is not None:
                    good_new_points = cur_points[st==1]
                    good_old_points = init_points[st==1]
                
                old_coords = []
                new_coords = []

                # Draw optical flow vectors and write out values
                for i, (new, old) in enumerate(zip(good_new_points, good_old_points)):
                    x0, y0 = old.ravel()
                    x1, y1 = new.ravel()
                    dx = x1 - x0
                    dy = y1 - y0

                    old_coords.append((x0, y0))
                    new_coords.append((x1, y1))
                    magnitudes.append(np.sqrt(dx**2 + dy**2))
                    angles.append(np.arctan2(dy, dx))

                if timestamp > rolling_time_win:
                    # store summary statistics
                    if output_detail_level == "summary":
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
                        num_points.append(len(good_new_points))
                        
                    else:
                        for i, (old,new) in enumerate(zip(old_coords, new_coords)):
                            sample_stats = []
                            sample_stats.extend([timestamp//1000, old[0], old[1], new[0], new[1], magnitudes[i], angles[i]])
                            full_stats.append(sample_stats)
                      
                    rolling_time_win += output_sample_frequency_msec

                # Update previous frame and points
                old_gray = gray_frame.copy()
                init_points = good_new_points.reshape(-1, 1, 2)

        capture.release()

        # Create and return dataframe
        if output_detail_level == "summary":
            output_df = pd.DataFrame({
                "Timestamp":timestamps,
                "Mean Magnitude":mean_magnitudes,
                "Deviation Magnitude":std_magnitudes,
                "Mean Angle":mean_angles,
                "Deviation Angle":std_angles,
                "Number of Points":num_points
            })
            
            outputs.update({f"{filename}":output_df})
        
        else:
            cols = ["Timestamp", "Old x", "Old y", "New x", "New y", "Magnitude", "Angle"]
            output_df = pd.DataFrame(full_stats, columns=cols)
            outputs.update({f"{filename}":output_df})
    
    return outputs