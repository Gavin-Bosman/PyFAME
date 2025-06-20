from pyfame.io import map_directory_structure, get_video_capture, get_video_writer, get_directory_walk
from pyfame.util.util_exceptions import *
from pyfame.timing.timing_curves import *
from pyfame.mesh.get_mesh_landmarks import *
from pyfame.layer import LayerPipeline, LayerConfig
import cv2 as cv
import os
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

### NOTE: think about adding functionality to handle multiple roi, and assigning each to a single manipulation
#### MORE PREVALENT NOTE: Set up some sort of mid-loop conditional checks on manipulations that cannot be transitioned, 
#### such as PLD, occlusion, or grid shuffle, where at onset/offset checkpoints, they specifically are passed a dt of zero.

def resolve_missing_timing(config:LayerConfig, video_duration:int) -> tuple[int,int]:
    onset = config.onset_t if config.onset_t is not None else 0
    offset = config.offset_t if config.offset_t is not None else video_duration
    return (onset, offset)

def apply_layers(layers:list[LayerConfig], input_dir:str, output_dir:str, with_sub_dirs:bool = False):
    """ Takes in a list of layer objects, and applies each manipulation layer in sequence frame-by-frame, and file-by-file for each file provided within input_dir.

    Parameters
    ----------

    layers: list of LayerConfig
        A list of LayerConfig wrappers containing the specified layer and its parameters.
    
    input_dir: str
        A path string to the directory containing all files to be processed.
    
    output_dir: str
        A path string to a directory where the processed files will be written to.
    
    with_sub_dirs: bool
        A boolean flag indicating if the input directory contains subdirectories.
    
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

    # Map the input dir structure to the output dir
    map_directory_structure(input_dir, output_dir, with_sub_dirs)
    if with_sub_dirs:
        sub_dirs, files_to_process = get_directory_walk(input_dir, with_sub_dirs)
    else:
        files_to_process = get_directory_walk(input_dir, with_sub_dirs)

    # Initialize the processing pipeline
    pipeline = LayerPipeline()
    for layer in layers:
        pipeline.add_layer(layer)

    for file in files_to_process:
        filename, extension = os.path.splitext(os.path.basename(file))
        static_image_mode = False
        codec = "mp4v"
        capture = None
        result = None
        cap_duration = 0
        dir_file_path = output_dir + f"\\{filename}_processed{extension}"

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                static_image_mode = False
            case ".mov":
                static_image_mode = False
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                static_image_mode = True
            case _:
                logger.error("Function has encountered an unparseable file type. " 
                             "Function exiting with status 1. Please see pyfameutils.transcode_video_to_mp4().")
                debug_logger.error(f"Function has encountered an unparseable file type {extension}. "
                                    "Consider using different input file formats, or transcoding video files with transcode_video_to_mp4().")
                raise UnrecognizedExtensionError()
        
        if not static_image_mode:
            capture = get_video_capture(file)
            size = (int(capture.get(3)), int(capture.get(4)))
            result = get_video_writer(dir_file_path, size, codec)

            # Getting the video duration for weight calculations
            frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv.CAP_PROP_FPS)
            cap_duration = float(frame_count)/float(fps)

            for config in pipeline.layers:
                resolve_missing_timing(config, cap_duration)
        
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
                output_frame = pipeline.apply_layers(output_frame)
                success = cv.imwrite(dir_file_path, output_frame)
                if not success:
                    raise FileWriteError()
                
                break
        
        if not static_image_mode:
            capture.release()
            result.release()