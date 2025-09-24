from pyfame.file_access import get_video_capture, get_video_writer, map_directory_structure, create_output_directory
from pyfame.utilities.exceptions import *
from pyfame.layer.timing_curves import *
from pyfame.mesh.mesh_landmarks import *
from pyfame.layer.layer import Layer
from pyfame.layer.layer_pipeline import LayerPipeline
from pyfame.logging.write_experiment_log import write_experiment_log
from pyfame.mesh.get_mesh_coordinates import get_mesh
import cv2 as cv
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def resolve_missing_timing(layer:Layer, video_duration:int) -> tuple[int,int]:
    onset = layer.onset_t if layer.onset_t is not None else 0
    offset = layer.offset_t if layer.offset_t is not None else video_duration

    # Update timing config object
    layer.config = layer.config.model_copy(update={
        "time_onset": onset,
        "time_offset": offset
    })

    # keep layer.self params in sync
    layer.onset_t = onset
    layer.offset_t = offset

    return onset, offset

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

    # Get the current sys timestamp for a unique output folder
    # name that will identify the processing session
    timestamp = datetime.now().isoformat(timespec='seconds')
    timestamp_str = timestamp.replace(":","-")

    # Ensure compatibility with Layer_Pipeline class
    if isinstance(layers, Layer):
        layers = [layers]

    # Extracting the i/o paths from the file_paths dataframe
    absolute_paths = file_paths["Absolute Path"]
    relative_paths = file_paths["Relative Path"]

    # Extracting the root directory name from the file paths
    norm_path = os.path.normpath(absolute_paths.iloc[0])
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
    output_directory = create_output_directory(output_directory, timestamp_str)
    # Map any scaffoled sub-organization from the input dir to the output dir
    map_directory_structure(input_directory, output_directory)

    # Initialize the processing pipeline
    pipeline = LayerPipeline()
    pipeline.add_layers(layers)

    static_image_mode = False
    face_mesh = get_mesh(
        min_tracking_confidence=layers[0].config.min_tracking_confidence, 
        min_detection_confidence=layers[0].config.min_detection_confidence,
        static_image_mode=static_image_mode,
        max_num_faces=1
    )

    # Iterate over file list
    for i,file in enumerate(
        tqdm(
            absolute_paths, 
            total=len(absolute_paths), 
            desc="Files processed:",
            bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}',
            position=0,
            dynamic_ncols=True
        )
    ):
        
        if not os.path.isfile(file):
            raise FileReadError(file_path=file)
        
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = "mp4v"
        capture = None
        result = None
        cap_duration = 0
        
        # Get the relative file path
        relative_file_path = relative_paths.iloc[i]
        
        subdirectory_names = [
            part for part in relative_file_path.split(os.sep)
            if part not in (root_directory, "raw", os.path.basename(file))
        ]

        dir_file_path = os.path.join(output_directory, *subdirectory_names, f"{filename}{extension}")

        # Using the file extension to sniff video codec or image container for images
        if str.lower(extension) not in {".mp4", ".mov", ".jpg", ".jpeg", ".png", ".bmp"}:
            print(f"Skipping unparseable file {os.path.basename(file)}.")
            continue
        # Reset the face mesh if switching between movies and images
        elif str.lower(extension) in {".jpg", ".jpeg", ".png", ".bmp"}:
            static_image_mode = True
            face_mesh = get_mesh(
                min_tracking_confidence=layers[0].config.min_tracking_confidence, 
                min_detection_confidence=layers[0].config.min_detection_confidence,
                static_image_mode=static_image_mode,
                max_num_faces=1
            )
        
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
        pb = tqdm(
            total=frame_count, 
            desc="Video frames processed:",
            bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', 
            colour="blue",
            position=1,
            dynamic_ncols=True
        )
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

            if not static_image_mode:
                # Getting the current video timestamp
                dt = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                output_frame = pipeline.apply_layers(face_mesh, frame, dt, file_path=file)
                output_frame = output_frame.astype(np.uint8)
                result.write(output_frame)
            else:
                output_frame = pipeline.apply_layers(face_mesh, frame, None, file_path=file)
                output_frame = output_frame.astype(np.uint8)
                success = cv.imwrite(dir_file_path, output_frame)
                if not success:
                    raise FileWriteError()
                
                break

            pb.update(1)
        
        pb.close()
        write_experiment_log(layers, file, test_path)
        
        if not static_image_mode:
            capture.release()
            result.release()
        
        for layer in tqdm(
            layers, 
            total=len(layers), 
            desc="Layer state reset:", 
            bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}',
            leave=False,
            colour="blue"
        ):
    
            # Reset back to initial state after user construction
            layer._reset_state()