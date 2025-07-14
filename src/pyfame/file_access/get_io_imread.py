import cv2 as cv
from pyfame.utilities.util_checks import *
from pyfame.utilities.util_exceptions import FileReadError

def get_io_imread(file_path:str, flag:int = cv.IMREAD_UNCHANGED) -> cv.typing.MatLike:
    check_type(file_path, [str])
    check_valid_path(file_path)
    check_is_file(file_path)
    check_type(flag, [int])
    
    img = cv.imread(file_path, flag)
    if img is None:
        raise FileReadError()
    else:
        return img