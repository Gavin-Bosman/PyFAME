from pyfame.file_access import *
from pyfame.utilities.checks import *
import os
import pandas as pd

def apply_conversion_video_to_mp4(file_paths:pd.DataFrame) -> None:
    """ Given an input directory containing one or more video files, transcodes all video files from their current
    container to mp4.

    Parameters
    ----------

    file_paths: DataFrame
        A 2-column dataframe consisting of absolute and relative file paths (relative to the working directory root).
    
    Raises
    ------

    FileReadError:
        If expected directory structure is missing, or input paths are invalid.
    
    Returns
    -------

    None
    
    """

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
    
    output_directory = os.path.join(test_path, "processed")

    for file in absolute_paths:
        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        capture = get_video_capture(file)
        size = (int(capture.get(3)), int(capture.get(4)))
        result = get_video_writer(file_path=os.path.join(output_directory, f"{filename}_transcoded.mp4"), frame_size=size)
        
        while True:
            success, frame = capture.read()
            if not success:
                break

            result.write(frame)

        capture.release()
        result.release()