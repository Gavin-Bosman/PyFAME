import cv2 as cv
import os
from pyfame.util import util_exceptions
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

def get_video_writer(dir_path:str, size:tuple[int,int], codec:str = 'mp4v', fps:int = 30, isColor:bool = True) -> cv.VideoWriter:
    """ Returns a cv2.VideoWriter instance, instantiated over the directory path provided.

    Parameters
    ----------

    file_path: str
        A path string to a directory.
    
    size: tuple of int
        The desired frame dimensions (in pixels) of the output video.
    
    codec: str
        The video codec used to encode an image sequence as a video file. 
    
    fps: int
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
    
    if not isinstance(dir_path, str):
        logger.error("Function encountered a TypeError for input parameter dir_path."
                     "Message: parameter dir_path expects a str.")
        raise TypeError("Get_video_writer: parameter dir_path expects a str.")
    elif not os.path.exists(dir_path):
        logger.error("Function encountered an OSError for input parameter dir_path."
                     "Message: parameter dir_path is not a valid path string.")
        raise OSError("Get_video_writer: parameter dir_path must be a valid path in the current scope.")
    elif not os.path.isdir(dir_path):
        logger.error("Function encountered a ValueError for input parameter dir_path. "
                     "Message: parameter dir_path must be a path string to a directory.")
        raise ValueError("Get_video_writer: parameter dir_path must be a path string to a directory.")
    
    if not isinstance(tuple, size):
        logger.error("Function encountered a TypeError for input parameter size."
                     "Message: parameter size expects a tuple of integers.")
        raise TypeError("Get_video_writer: parameter size expects a tuple of integers.")
    for elem in size:
        if not isinstance(elem, int):
            logger.error("Function encountered a TypeError for input parameter size."
                         "Message: parameter size expects a tuple of integers.")
            raise TypeError("Get_video_writer: parameter size expects a tuple of integers.")
    
    if not isinstance(codec, str):
        logger.error("Function encountered a TypeError for input parameter codec."
                     "Message: parameter codec expects a str.")
        raise TypeError("Get_video_writer: parameter codec expects a str.")
    
    if not isinstance(fps, int):
        logger.error("Function encountered a TypeError for input parameter fps. "
                     "Message: parameter fps expects an integer.")
        raise TypeError("Get_video_writer: parameter fps expects an integer.")
    
    if not isinstance(isColor, bool):
        logger.error("Function encountered a TypeError for input parameter isColor. "
                     "Message: parameter isColor expects a boolean.")
        raise TypeError("Get_video_writer: parameter isColor expects a bool.")


    vw = cv.VideoWriter(dir_path, cv.VideoWriter.fourcc(*codec), fps, size, isColor=isColor)

    if not vw.isOpened():
        raise util_exceptions.FileWriteError("Function encountered an error attempting to instantiate "
                                        f"cv.VideoWriter() over file {dir_path}.")
    else:
        return vw