import cv2 as cv
import os
from pyfame.util import util_exceptions
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def get_video_capture(file_path:str) -> cv.VideoCapture:
    """ Returns a cv2.VideoCapture instance instantiated over the file found at the provided path string.
    
    Parameters
    ----------

    file_path: str
        A path string to a video file.
    
    Raises
    ------

    TypeError

    ValueError

    OSError

    UnrecognizedExtensionError

    FileReadError

    Returns
    -------

    cv2.VideoCapture
    """

    if not isinstance(file_path, str):
        logger.error("Function encountered a TypeError for input parameter file_path."
                    "Message: parameter file_path expects a str.")
        raise TypeError("Create_output_directory: parameter file_path expects a str.")
    elif not os.path.exists(file_path):
        logger.error("Function encountered an OSError for input parameter file_path."
                    "Message: parameter file_path is not a valid path string.")
        raise OSError("Create_output_directory: parameter file_path must be a valid directory path in the current scope.")
    elif os.path.isdir(file_path):
        logger.error("Function encountered a ValueError for input parameter file_path."
                    "Message: parameter file_path must be a path string to a video file, not a directory.")
        raise ValueError("Create_output_directory: parameter file_path must contain a path string to a video file, not a directory.")
    
    filename, extension = os.path.splitext(os.path.basename(file_path))
    if extension not in [".mp4", ".mov"]:
        logger.error("Function has encountered an UnrecognizedExtensionError on input parameter file_path. "
                     "Message: file_path must be a path string pointing to a video file (.mp4, .mov).")
        raise util_exceptions.UnrecognizedExtensionError(extension=extension)

    vc = cv.VideoCapture(file_path)

    if not vc.isOpened():
        raise util_exceptions.FileReadError("Function has encountered an error attempting to instantiate cv2.VideoCapture()"
                                       f" over file {file_path}.")
    else:
        return vc