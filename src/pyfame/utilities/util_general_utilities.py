from pyfame.utilities.util_constants import *
from math import atan
import numpy as np

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

    angle = (slope_2-slope_1) / (1 + slope_1*slope_2)
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