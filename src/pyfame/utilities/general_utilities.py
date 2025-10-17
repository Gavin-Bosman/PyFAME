from pyfame.utilities.constants import *
from pyfame.landmark import facial_landmarks
from math import atan
import numpy as np
import cv2 as cv

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

def get_landmark_names(landmarks, module_globals = vars(facial_landmarks)):
    """ Takes in a list of landmark paths, and returns its internal variable name. 
    """
    # check for sublists
    if isinstance(landmarks[0], list):
        names = []
        for i,lm in enumerate(landmarks, start=1):
            for name, val in module_globals.items():
                if isinstance(val, list) and val == lm:
                    names.append(name)
            
            if len(names) < i:
                names.append("UNKNOWN_LANDMARK")
        
        return names
    
    else:
        for name, val in module_globals.items():
            if isinstance(val, list) and val == landmarks:
                return name
        
        return "UNKNOWN_LANDMARK"

def compute_rotation_angle(slope_1:float, slope_2:float = 0.0) -> float:
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
    """ Given two integer points in the form (x,y), compute the slope of the line 
    that connects the two points and return it.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx == 0:
        return float("inf")
    return dy/dx

def display_landmarks_face_overlay(frame, landmarker_coordinates, point_radius:int = 4):
    """ Given a frame or image containing a face, and the FaceLandmarker landmark coordinates list
    returned from applying the FaceLandmarker to the frame or image, visualize all landmark points on the 
    frame or image.
    """
    for (x, y) in landmarker_coordinates:
        cv.circle(frame, (x,y), point_radius, (0, 0, 255), -1)