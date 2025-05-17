import cv2 as cv
from pyfame.utils import exceptions

def get_video_writer(file_path:str, size:tuple[int,int], codec:str = 'mp4v', fps:int = 30) -> cv.VideoWriter:
    vw = cv.VideoWriter(file_path, cv.VideoWriter.fourcc(*codec), fps, size)

    if not vw.isOpened():
        raise exceptions.FileWriteError("Function encountered an error attempting to instantiate "
                                        f"cv.VideoWriter() over file {file_path}.")
    else:
        return vw