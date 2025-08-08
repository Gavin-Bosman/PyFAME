import cv2 as cv
from pyfame.utilities.checks import *
from pyfame.utilities.exceptions import FileReadError

def get_imread(file_path:str, flag:int = cv.IMREAD_UNCHANGED) -> cv.typing.MatLike:
    # Perform parameter checks
    check_type(file_path, [str])
    check_valid_path(file_path)
    check_is_file(file_path)
    check_type(flag, [int])
    
    # Read in the image, check for errors before returning
    img = cv.imread(file_path, flag)
    if img is None:
        raise FileReadError()
    else:
        return img