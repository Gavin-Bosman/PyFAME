import cv2 as cv
from pyfame.utils import exceptions

def get_video_capture(file_path:str) -> cv.VideoCapture:
    vc = cv.VideoCapture(file_path)

    if not vc.isOpened():
        raise exceptions.FileReadError("Function has encountered an error attempting to instantiate cv2.VideoCapture()"
                                       f" over file {file_path}.")
    else:
        return vc