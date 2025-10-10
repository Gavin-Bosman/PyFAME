from pyfame.utilities.constants import *
from pyfame.landmark import facial_landmarks
from math import atan
import numpy as np
import cv2 as cv

def display_landmarks_face_overlay(frame, lm_screen_coords, point_radius:int = 4):
    for (x, y) in lm_screen_coords:
        cv.circle(frame, (x,y), point_radius, (0, 0, 255), -1)

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

def sanitize_json_value(value):
    """ Takes in a value of any type, and converts it to a JSON-serializable type
    then returns it.

    parameters
    ----------

    value: any
        The value to be sanitized.
    
    returns
    -------

    sanitized_value: any
        The input value converted to a compatible, JSON-serializable type.
    """
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif callable(value):
        return value.__name__
    elif isinstance(value, dict):
        return {k:sanitize_json_value(v) for k,v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [sanitize_json_value(v) for v in value]
    else:
        return value

def get_landmark_name(landmark, module_globals = vars(facial_landmarks)):
    """ Takes in a region of interest list, and returns its internal variable name. 
        
    """
    # check for sublists
    if isinstance(landmark[0], list):
        names = []
        for i,roi in enumerate(landmark, start=1):
            for name, val in module_globals.items():
                if isinstance(val, list) and val == roi:
                    names.append(name)
            
            if len(names) < i:
                names.append("UNKNOWN_ROI")
        
        return names
    
    else:
        for name, val in module_globals.items():
            if isinstance(val, list) and val == landmark:
                return name
        
        return "UNKNOWN_ROI"

def compute_rot_angle(slope_1:float, slope_2:float = 0.0) -> float:
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

    rot_angle: float
        The displacement between the two slopes. 
    """

    if np.isinf(slope_1) and np.isinf(slope_2):
        return 0.0
    elif np.isinf(slope_1):
        return 90.0
    elif np.isinf(slope_2):
        return -90.0

    rad_angle = atan((slope_2-slope_1) / (1 + slope_1*slope_2))
    rot_angle = np.degrees(rad_angle)
    return rot_angle

def compute_slope(p1:tuple[int,int], p2:tuple[int,int]) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx == 0:
        return float("inf")
    return dy/dx

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