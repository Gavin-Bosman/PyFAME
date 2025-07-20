from pyfame.file_access import get_video_capture, get_video_writer, get_directory_walk, contains_sub_directories, create_output_directory
from pyfame.utilities.exceptions import *
from pyfame.layer.timing_curves import *
from pyfame.mesh.mesh_landmarks import *
from .layer import Layer
from .layer_pipeline import LayerPipeline
from ..logging.write_experiment_log import write_experiment_log
import cv2 as cv
import os
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

### NOTE: Add some method of enforcing strict ordering with manipulations that affect the entire image, rather than just a facial region

def resolve_missing_timing(layer:Layer, video_duration:int) -> tuple[int,int]:
    onset = layer.onset_t if layer.onset_t is not None else 0
    offset = layer.offset_t if layer.offset_t is not None else video_duration
    return (onset, offset)

def apply_layers(layers:list[Layer]):
    """ Takes in a list of layer objects, and applies each manipulation layer in sequence frame-by-frame, and file-by-file for each file provided within input_dir.

    Parameters
    ----------

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
    
    files_to_process = []
    sub_dirs = []

    ### TODO need some way of permanently storing the named root output folder
    cwd = os.getcwd()
    test_path = os.path.join(cwd, "data")

    if not os.path.isdir(test_path):
        raise FileReadError(message="Unable to locate the input 'data/' directory. Please call make_output_paths() to set up the output directory.")

    input_directory = os.path.join(cwd, "data/raw")
    output_directory = os.path.join(cwd, "data/results")

    if contains_sub_directories(input_directory):
        sub_dirs, files_to_process = get_directory_walk(input_directory, True)
    else:
        files_to_process = get_directory_walk(input_directory, False)
    
    if len(sub_dirs) >= 1:
        sub_dirs = [os.path.basename(path) for path in sub_dirs]
    
    # Initialize the processing pipeline
    pipeline = LayerPipeline()
    pipeline.add_layers(layers)

    for file in files_to_process:
        if not os.path.isfile(file):
            raise FileReadError(file_path=file)
        
        filename, extension = os.path.splitext(os.path.basename(file))
        static_image_mode = False
        codec = "mp4v"
        capture = None
        result = None
        cap_duration = 0

        pre_path, *_ = os.path.split(file)
        dir_name = os.path.basename(pre_path)

        if dir_name in sub_dirs:
            create_output_directory(output_directory, dir_name)
            dir_file_path = os.path.join(output_directory, dir_name, f"{filename}_processed{extension}")
        else:
            dir_file_path = os.path.join(output_directory, f"{filename}_processed{extension}")

        # Using the file extension to sniff video codec or image container for images
        match str.lower(extension):
            case ".mp4":
                static_image_mode = False
            case ".mov":
                static_image_mode = False
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                static_image_mode = True
            case _:
                raise UnrecognizedExtensionError(extension=extension)
        
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
            output_frame = frame.copy()

            if not static_image_mode:
                # Getting the current video timestamp
                dt = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                output_frame = pipeline.apply_layers(output_frame, dt)
                result.write(output_frame)
            else:
                output_frame = pipeline.apply_layers(output_frame, dt, static_image_mode=True)
                success = cv.imwrite(dir_file_path, output_frame)
                if not success:
                    raise FileWriteError()
                
                break
        
        write_experiment_log(layers, file)
        
        if not static_image_mode:
            capture.release()
            result.release()