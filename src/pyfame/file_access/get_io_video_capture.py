import cv2 as cv
import os
from pyfame.utilities.util_exceptions import *
from pyfame.utilities.util_checks import *
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

    check_type(file_path, [str])
    check_valid_path(file_path)
    check_is_dir(file_path)
    
    filename, extension = os.path.splitext(os.path.basename(file_path))
    check_valid_file_extension(extension=extension, allowed_extensions=[".mp4", ".mov"])

    vc = cv.VideoCapture(file_path)

    if not vc.isOpened():
        raise FileReadError("Function has encountered an error attempting to instantiate cv2.VideoCapture()"
                           f" over file {file_path}.")
    else:
        return vc