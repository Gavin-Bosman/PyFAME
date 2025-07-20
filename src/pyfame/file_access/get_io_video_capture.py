import cv2 as cv
import os
from pyfame.utilities.exceptions import *
from pyfame.utilities.checks import *
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def get_video_capture(file_path:str) -> cv.VideoCapture:
    # Perform parameter checks
    check_type(file_path, [str])
    check_valid_path(file_path)
    check_is_dir(file_path)
    
    # Isolate the file extension to verify its compatibility with cv2.VideoCapture
    filename, extension = os.path.splitext(os.path.basename(file_path))
    check_valid_file_extension(extension=extension, allowed_extensions=[".mp4", ".mov"])

    # Instantiate the VideoCapture, then check for errors before returning the instance
    vc = cv.VideoCapture(file_path)

    if not vc.isOpened():
        raise FileReadError("Function has encountered an error attempting to instantiate cv2.VideoCapture()"
                           f" over file {file_path}.")
    else:
        return vc