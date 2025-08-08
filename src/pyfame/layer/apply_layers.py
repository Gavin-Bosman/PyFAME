from pyfame.file_access import get_video_capture, get_video_writer, map_directory_structure
from pyfame.utilities.exceptions import *
from pyfame.layer.timing_curves import *
from pyfame.mesh.mesh_landmarks import *
from .layer import Layer
from .layer_pipeline import LayerPipeline
from pyfame.logging.write_experiment_log import write_experiment_log
import cv2 as cv
import os
import pandas as pd

### TODO: consider adding a state-reset method to the layers, or call __init__ after each file processing is complete

def resolve_missing_timing(layer:Layer, video_duration:int) -> tuple[int,int]:
    onset = layer.onset_t if layer.onset_t is not None else 0
    offset = layer.offset_t if layer.offset_t is not None else video_duration
    return (onset, offset)

def apply_layers(file_paths:pd.DataFrame, layers:list[Layer] | Layer):
    """ Takes in a list of layer objects, and applies each manipulation layer in sequence frame-by-frame, and file-by-file for each file provided within input_dir.

    Parameters
    ----------

    file_paths: pandas.DataFrame
        A table of path strings returned by 
    
    layers: list of Layer
        A list of Layer objects containing the specified layer and its parameters.
    
    Raises
    ------

    TypeError

    ValueError
    
    OSError

    FileReadError

    FileWriteError

    UnrecognizedExtensionError
    
    Returns
    -------

    None
    """

    # Ensure compatibility with Layer_Pipeline class
    if isinstance(layers, Layer):
        layers = [layers]

    # Extracting the i/o paths from the file_paths dataframe
    absolute_paths = file_paths["Absolute Path"]
    relative_paths = file_paths["Relative Path"]

    # Extracting the root directory name from the file paths
    norm_path = os.path.normpath(absolute_paths[0])
    norm_cwd = os.path.normpath(os.getcwd())
    rel_dir_path, *_ = os.path.split(os.path.relpath(norm_path, norm_cwd))
    parts = rel_dir_path.split(os.sep)
    root_directory = None

    if parts is not None:
        root_directory = parts[0]
    
    if root_directory is None:
        root_directory = "data"
    
    # Test that the directory provided actually exists in the file system before any read/writes
    test_path = os.path.join(norm_cwd, root_directory)

    if not os.path.isdir(test_path):
        raise FileReadError(message=f"Unable to locate the input {root_directory} directory. Please call make_output_paths() to set up the correct directory structure.")
    if not os.path.isdir(os.path.join(test_path, "raw")):
        raise FileReadError(message=f"Unable to locate the 'raw' subdirectory under root directory '{root_directory}'. Please call make_output_paths() to set up the correct directory structure.")
    if not os.path.isdir(os.path.join(test_path, "processed")):
        raise FileReadError(message=f"Unable to locate the 'processed' subdirectory under root directory '{root_directory}'. Please call make_output_paths() to set up the correct directory structure.")

    # Pre-made subdirectory structure in the project root
    input_directory = os.path.join(test_path, "raw")
    output_directory = os.path.join(test_path, "processed")
    # Map any scaffoled sub-organization from the input dir to the output dir
    map_directory_structure(input_directory, output_directory)

    # Initialize the processing pipeline
    pipeline = LayerPipeline()
    pipeline.add_layers(layers)

    for i,file in enumerate(absolute_paths):
        if not os.path.isfile(file):
            raise FileReadError(file_path=file)
        
        filename, extension = os.path.splitext(os.path.basename(file))
        static_image_mode = False
        codec = "mp4v"
        capture = None
        result = None
        cap_duration = 0
        
        # Get the relative file path
        relative_file_path = relative_paths[i]
        
        subdirectory_names = [
            part for part in relative_file_path.split(os.sep)
            if part not in (root_directory, "raw", os.path.basename(file))
        ]

        dir_file_path = os.path.join(output_directory, *subdirectory_names, f"{filename}_processed{extension}")

        # Using the file extension to sniff video codec or image container for images
        if str.lower(extension) not in [".mp4", ".mov", ".jpg", ".jpeg", ".png", ".bmp"]:
            print(f"Skipping unparseable file {os.path.basename(file)}.")
            continue
        
        if not static_image_mode:
            capture = get_video_capture(file)
            size = (int(capture.get(3)), int(capture.get(4)))
            result = get_video_writer(dir_file_path, size, codec)

            # Getting the video duration for weight calculations
            frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv.CAP_PROP_FPS)

            if fps == 0:
                raise FileReadError(message="Input video fps is zero. File may be corrupt or incorrectly encoded.")
            else:
                cap_duration = float(frame_count)/float(fps)

            for layer in pipeline.layers:
                resolve_missing_timing(layer, cap_duration)
        
        # Loop over the current file until completion; (single iteration for static images)
        while(True):
            frame = None
            if static_image_mode:
                frame = cv.imread(file)
                if frame is None:
                    raise FileReadError()
            else:
                success, frame = capture.read()
                if not success:
                    break
            
            # declaring variables so they maintain their larger scope
            dt = None

            if not static_image_mode:
                # Getting the current video timestamp
                dt = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                output_frame = pipeline.apply_layers(frame, dt)
                output_frame = output_frame.astype(np.uint8)
                result.write(output_frame)
            else:
                output_frame = pipeline.apply_layers(frame, dt, static_image_mode=True)
                output_frame = output_frame.astype(np.uint8)
                success = cv.imwrite(dir_file_path, output_frame)
                if not success:
                    raise FileWriteError()
                
                break
        
        write_experiment_log(layers, file, test_path)
        
        if not static_image_mode:
            capture.release()
            result.release()