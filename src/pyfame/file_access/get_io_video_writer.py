import cv2 as cv
import os
from pyfame.utilities.util_exceptions import *
from pyfame.utilities.util_checks import *
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def get_video_writer(file_path:str, frame_size:tuple[int,int], video_codec:str = 'mp4v', frame_rate:int = 30, isColor:bool = True) -> cv.VideoWriter:
    """ Returns a cv2.VideoWriter instance, instantiated over the directory path provided.

    Parameters
    ----------

    file_path: str
        A path string to an output file.
    
    frame_size: tuple of int
        The desired frame dimensions (in pixels) of the output video.
    
    video_codec: str
        The video codec used to encode an image sequence as a video file. 
    
    frame_rate: int
        The desired frame rate (in frames/second) of the output video.
    
    isColor: bool
        A boolean flag indicating if the output video file will be written in full color.
    
    Raises
    ------

    TypeError

    ValueError

    OSError

    FileWriteError

    Returns
    -------

    cv2.VideoWriter
    """
    
    check_type(file_path, [str])
    check_valid_path(file_path)
    check_is_file(file_path)

    check_type(frame_size, [tuple])
    check_type(frame_size, [int], iterable=True)

    check_type(video_codec, [str])
    check_value(video_codec, ['mp4v', 'XVID'])

    check_type(frame_rate, [int])
    check_value(frame_rate, min=1, max=60)

    check_type(isColor, [bool])

    vw = cv.VideoWriter(file_path, cv.VideoWriter.fourcc(*video_codec), frame_rate, frame_size, isColor=isColor)

    if not vw.isOpened():
        raise FileWriteError("Function encountered an error attempting to instantiate "
                            f"cv.VideoWriter() over file {file_path}.")
    else:
        return vw