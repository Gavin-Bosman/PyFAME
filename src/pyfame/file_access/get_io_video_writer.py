import cv2 as cv
import os
from pyfame.utilities.exceptions import *
from pyfame.utilities.checks import *
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def get_video_writer(file_path:str, frame_size:tuple[int,int], video_codec:str = 'mp4v', frame_rate:int = 30, isColor:bool = True) -> cv.VideoWriter:
    # Perform parameter checks
    check_type(file_path, [str])
    check_valid_path(file_path)

    check_type(frame_size, [tuple])
    check_type(frame_size, [int], iterable=True)

    check_type(video_codec, [str])
    check_value(video_codec, ['mp4v', 'XVID'])

    check_type(frame_rate, [int])
    check_value(frame_rate, min=1, max=60)

    check_type(isColor, [bool])

    # Create VideoWriter instance
    vw = cv.VideoWriter(file_path, cv.VideoWriter.fourcc(*video_codec), frame_rate, frame_size, isColor=isColor)
    # Check for any errors with object creation before returning the videoWriter instance
    if not vw.isOpened():
        raise FileWriteError("Function encountered an error attempting to instantiate "
                            f"cv.VideoWriter() over file {file_path}.")
    else:
        return vw