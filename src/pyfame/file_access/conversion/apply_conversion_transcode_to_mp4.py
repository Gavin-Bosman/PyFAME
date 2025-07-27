from pyfame.file_access import *
from pyfame.utilities.checks import *
import os

def apply_conversion_video_to_mp4(input_directory:str, output_directory:str, with_sub_dirs:bool = False) -> None:
    """ Given an input directory containing one or more video files, transcodes all video files from their current
    container to mp4. This function can be used to preprocess older video file types before masking, occluding or color shifting.

    Parameters
    ----------

    input_directory: str
        A path string to the directory containing the videos to be transcoded.
    
    output_directory: str
        A path string to the directory where transcoded videos will be written too.
    
    with_sub_dirs: bool
        A boolean flag indicating if the input directory contains sub-directories.
    
    Raises
    ------

    TypeError
        Given invalid parameter types.
    OSError
        Given invalid paths for input_dir or output_dir.
    
    Returns
    -------

    None
    
    """

    # Type checking input parameters
    check_type(input_directory, [str])
    check_valid_path(input_directory)

    check_type(output_directory, [str])
    check_valid_path(output_directory)

    check_type(with_sub_dirs, [bool])

    files_df = get_directory_walk(input_directory)
    files_to_process = files_df["Absolute Path"]
    
    for file in files_to_process:
        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        capture = get_video_capture(file)
        size = (int(capture.get(3)), int(capture.get(4)))
        result = get_video_writer(output_directory + "\\" + filename + "_transcoded.mp4", size)
        
        while True:
            success, frame = capture.read()
            if not success:
                break

            result.write(frame)

        capture.release()
        result.release()