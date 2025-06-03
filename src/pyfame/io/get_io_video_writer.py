import cv2 as cv
from pyfame.util import util_exceptions

def get_video_writer(file_path:str, size:tuple[int,int], codec:str = 'mp4v', fps:int = 30, isColor:bool = True) -> cv.VideoWriter:
    vw = cv.VideoWriter(file_path, cv.VideoWriter.fourcc(*codec), fps, size, isColor=isColor)

    if not vw.isOpened():
        raise util_exceptions.FileWriteError("Function encountered an error attempting to instantiate "
                                        f"cv.VideoWriter() over file {file_path}.")
    else:
        return vw