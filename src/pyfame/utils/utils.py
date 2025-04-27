from .predefined_constants import *
from .landmarks import *
from pyfame.core.exceptions import FileReadError, FileWriteError
from math import atan
import numpy as np
import pandas as pd
import cv2 as cv
import os

def get_variable_name(variable, namespace) -> str:
    """ Takes in a variable and a namespace (one of locals() or globals()), and returns the variables defined name in the 
        relevant scope.

        parameters
        ----------

        variable: any
            Any defined variable.
        
        namespace: function
            One of locals() or globals(), defines which scope for the function to look at when searching for the variables name.
        
        returns
        -------

        variable_name: str
            The assigned variable name at the given scope level (if any).
    """

    return [name for name, value in namespace.items() if value == variable][0]

def compute_rot_angle(slope1:float, slope2:float = 0.0) -> float:
    """ Given the current and previous slope, this function uses the arctan function to compute the angle delta 
    in radians. If no previous slope is provided, the default value is zero.

    Parameters
    ----------

    slope1: float
        The current slope.

    slope2: float
        The initial slope (defaults to 0.0)

    returns
    -------

    rot_angle:float
        The displacement between the two slopes, provided in radians. 
    """

    angle = abs((slope2-slope1) / (1 + slope1*slope2))
    rad_angle = atan(angle)
    rot_angle = (rad_angle * 180) / np.pi
    return rot_angle

def compute_line_intersection(p1:tuple, p2:tuple, line:int, vertical:bool=False) -> tuple | None:
    """ compute_line_intersection takes two (x,y) points, and a line. If the path of the two provided points intersects the 
    provided line, an intersection point (x,y) is calculated and returned.

    Parameters:
    -----------

    p1: tuple of int
        The first (x,y) point to be compared.
    
    p2: tuple of int
        The second (x,y) point to be compared.
    
    line: int
        An integer representing a line in the x-y coordinate space.
    
    vertical: bool
        A boolean flag indicating whether or not the comparison line is vertical.
    
    Raises:
    -------

    TypeError:
        Given invalid parameter types.
    ValueError:
        Given unrecognized parameter types.

    Returns:
    --------

    A point (x,y) intersecting the provided line, or None if no such point exists.
    """

    # Performing type checks on input params
    if not isinstance(p1, tuple):
        raise TypeError("Function encountered a TypeError for input parameter p1. "
                        "Message: parameter p1 must be a tuple of integers.")
    elif not isinstance(p1[0], int) or not isinstance(p1[1], int):
        raise TypeError("Function encountered a TypeError for input parameter p1. "
                        "Message: parameter p1 must be a tuple of integers.")
    elif p1[0] < 0 or p1[1] < 0:
        raise ValueError("Function encountered a ValueError for input parameter p1. "
                         "Message: pixel coordinates must be positive integers.")
    
    if not isinstance(p2, tuple):
        raise TypeError("Function encountered a TypeError for input parameter p2. "
                        "Message: parameter p1 must be a tuple of integers.")
    elif not isinstance(p2[0], int) or not isinstance(p2[1], int):
        raise TypeError("Function encountered a TypeError for input parameter p2. "
                        "Message: parameter p1 must be a tuple of integers.")
    elif p2[0] < 0 or p2[1] < 0:
        raise ValueError("Function encountered a ValueError for input parameter p2. "
                         "Message: pixel coordinates must be positive integers.")
    
    if not isinstance(line, int):
        raise TypeError("Function enountered a TypeError for input parameter line. "
                        "Message: parameter line must be an integer. ")

    x1, y1 = p1
    x2, y2 = p2

    if vertical:
        if (x1 < line and x2 >= line) or (x1 >= line and x2 < line):
            # Calculate intersection point
            t = (line - x1) / (x2 - x1)
            intersect_y = y1 + t * (y2-y1)
            return (line, round(intersect_y))
    else:
        if (y1 < line and y2 >= line) or  (y1 >= line and y2 < line):
            # Calculate intersection point
            t = (line - y1) / (y2 - y1)
            intersect_x = x1 + t * (x2-x1)
            return (round(intersect_x), line)
    
    return None

def get_min_max_bgr(filePath:str, focusColor:int|str = COLOR_RED) -> tuple:
    """Given an input video file path, returns the minimum and maximum (B,G,R) colors, containing the minimum and maximum
    values of the focus color. 
    
    Parameters
    ----------

    filePath: str
        The path string of the location of the file to be processed.
    
    focusColor: int, str
        The RGB color channel to focus on. Either one of the predefined color constants, or a string literal of the colors name.
        
    Raises
    ------
    
    TypeError 
        Given invalid parameter types.
    ValueError 
        Given a nonexisting file path, or a non RGB focus color.
        
    Returns
    -------

    min_color: array of int
        A BGR colour code (ie. (100, 105, 80)) containing the minimum value of the focus color.
    max_color: array of int
        A BGR colour code (ie. (100, 105, 80)) containing the minimum value of the focus color.
    """

    # Type and value checking before computation
    if not isinstance(filePath, str):
        raise TypeError("get_min_max_rgb: invalid type for filePath.")
    elif not os.path.exists(filePath):
        raise ValueError("get_min_max_rgb: filePath not a valid path.")
    
    if isinstance(focusColor, str):
        if str.lower(focusColor) not in ["red", "green", "blue"]:
            raise ValueError("get_min_max_rgb: focusColor not a valid color option.")
    elif isinstance(focusColor, int):
        if focusColor not in [COLOR_RED, COLOR_BLUE, COLOR_GREEN]:
            raise ValueError("get_min_max_rgb: focusColor not a valid color option.")
    else:
        raise TypeError("get_min_max_rgb: invalid type for focusColor.")

    capture = cv.VideoCapture(filePath)
    if not capture.isOpened():
        print("get_min_max_rgb: Error opening videoCapture object.")
        return -1

    min_x, min_y, max_x, max_y, min_color, max_color = 0,0,0,0,None,None
    min_val, max_val = 255, 0

    while True:

        success, frame = capture.read()
        if not success:
            break

        blue, green, red = cv.split(frame)

        if focusColor == COLOR_RED or str.lower(focusColor) == "red":
            max_y = np.where(red == red.max())[0][0]
            max_x = np.where(red == red.max())[1][0]
            cur_max_val = red[max_y, max_x]

            min_y = np.where(red == red.min())[0][0]
            min_x = np.where(red == red.min())[1][0]
            cur_min_val = red[min_y, min_x]

            if cur_max_val > max_val:
                max_val = cur_max_val
                max_color = frame[max_y, max_x]

            if cur_min_val < min_val:
                min_val = cur_min_val
                min_color = frame[min_y, min_x]

        elif focusColor == COLOR_BLUE or str.lower(focusColor) == "blue":
            max_y = np.where(blue == blue.max())[0][0]
            max_x = np.where(blue == blue.max())[1][0]
            cur_max_val = blue[max_y, max_x]

            min_y = np.where(blue == blue.min())[0][0]
            min_x = np.where(blue == blue.min())[1][0]
            cur_min_val = blue[min_y, min_x]

            if cur_max_val > max_val:
                max_val = cur_max_val
                max_color = frame[max_y, max_x]

            if cur_min_val < min_val:
                min_val = cur_min_val
                min_color = frame[min_y, min_x]
        
        else:
            max_y = np.where(green == green.max())[0][0]
            max_x = np.where(green == green.max())[1][0]
            cur_max_val = green[max_y, max_x]

            min_y = np.where(green == green.min())[0][0]
            min_x = np.where(green == green.min())[1][0]
            cur_min_val = green[min_y, min_x]

            if cur_max_val > max_val:
                max_val = cur_max_val
                max_color = frame[max_y, max_x]

            if cur_min_val < min_val:
                min_val = cur_min_val
                min_color = frame[min_y, min_x]
    
    return (min_color, max_color)

def transcode_video_to_mp4(input_dir:str, output_dir:str, with_sub_dirs:bool = False) -> None:
    """ Given an input directory containing one or more video files, transcodes all video files from their current
    container to mp4. This function can be used to preprocess older video file types before masking, occluding or color shifting.

    Parameters
    ----------

    input_dir: str
        A path string to the directory containing the videos to be transcoded.
    
    output_dir: str
        A path string to the directory where transcoded videos will be written too.
    
    with_sub_dirs: bool
        A boolean flag indicating if the input directory contains sub-directories.
    
    Raises
    ------

    TypeError
        Given invalid parameter types.
    OSError
        Given invalid paths for input_dir or output_dir.
    """

    # Type checking input parameters
    singleFile = False
    if not isinstance(input_dir, str):
        raise TypeError("Transcode_video: parameter input_dir must be of type str.")
    if not os.path.exists(input_dir):
        raise OSError("Transcode_video: parameter input_dir must be a valid path string.")
    elif os.path.isfile(input_dir):
        singleFile = True

    if not isinstance(output_dir, str):
        raise TypeError("Transcode_video: parameter output_dir must be of type str.")
    if not os.path.exists(output_dir):
        raise OSError("Transcode_video: parameter output_dir must be a valid path string.")
    
    if not isinstance(with_sub_dirs, bool):
        raise TypeError("Transcode_video: parameter with_sub_dirs must be of type bool.")

    files_to_process = []
    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    for file in files_to_process:
        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        capture = cv.VideoCapture(file)
        if not capture.isOpened():
            print("Transcode_video: Error opening VideoCapture object.")
            raise FileReadError()
        
        size = (int(capture.get(3)), int(capture.get(4)))
        result = cv.VideoWriter(output_dir + "\\" + filename + "_transcoded.mp4",
                                cv.VideoWriter.fourcc(*'MP4V'), 30, size)
        if not result.isOpened():
            print("Transcode_video: Error opening VideoWriter object.")
            raise FileWriteError()
        
        while True:
            success, frame = capture.read()
            if not success:
                break

            result.write(frame)

        capture.release()
        result.release()

def create_path(landmark_set:list[int]) -> list[tuple]:
    """Given a list of facial landmarks (int), returns a list of tuples, creating a closed path in the form 
    [(a,b), (b,c), (c,d), ...]. This function allows the user to create custom facial landmark sets, for use in 
    mask_face_region() and occlude_face_region().
    
    Parameters
    ----------

    landmark_set: list of int
        A python list containing facial landmark indicies.
    
    Returns
    -------
        
    closed_path: list of tuple
        A list of tuples containing overlapping points, forming a path.
    """
    
    # Connvert the input list to a two-column dataframe
    landmark_dataframe = pd.DataFrame([(landmark_set[i], landmark_set[i+1]) for i in range(len(landmark_set) - 1)], columns=['p1', 'p2'])
    closed_path = []

    # Initialise the first two points
    p1 = landmark_dataframe.iloc[0]['p1']
    p2 = landmark_dataframe.iloc[0]['p2']

    for i in range(0, landmark_dataframe.shape[0]):
        obj = landmark_dataframe[landmark_dataframe['p1'] == p2]
        p1 = obj['p1'].values[0]
        p2 = obj['p2'].values[0]

        current_route = (p1, p2)
        closed_path.append(current_route)
    
    return closed_path