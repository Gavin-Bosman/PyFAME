import cv2 as cv
import pandas as pd
import numpy as np
import os
import sys
from math import atan

# Defining pertinent facemesh landmark sets
FACE_OVAL_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 366, 401, 288, 397, 365, 379, 378, 400, 377, 
                 152, 148, 176, 149, 150, 136, 172, 58, 177, 137, 234, 127, 162, 21, 54, 103, 67, 109, 10]
FACE_OVAL_TIGHT_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 345, 352, 376, 433, 397, 365, 379, 378, 400, 377, 
            152, 148, 176, 149, 150, 136, 172, 213, 147, 123, 116, 127, 162, 21, 54, 103, 67, 109, 10]

LEFT_EYE_IDX = [301, 298, 333, 299, 336, 285, 413, 464, 453, 452, 451, 450, 449, 448, 261, 265, 383, 301]
LEFT_IRIS_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE_IDX = [71, 68, 104, 69, 107, 55, 189, 244, 233, 232, 231, 230, 229, 228, 31, 35, 156, 71]
RIGHT_IRIS_IDX = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263]

NOSE_IDX = [168, 193, 122, 196, 174, 217, 209, 49, 129, 64, 98, 167, 164, 393, 327, 294, 278, 279, 429, 437, 
            399, 419, 351, 417, 168]
LIPS_IDX = [164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 186, 92, 165, 167, 164]
LIPS_TIGHT_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 0, 37, 39, 40, 185, 61]
LEFT_CHEEK_IDX = [207, 187, 147, 123, 116, 111, 31, 228, 229, 230, 231, 232, 233, 188, 196, 174, 217, 209, 49, 203, 206, 207]
RIGHT_CHEEK_IDX = [427, 411, 376, 352, 345, 340, 261, 448, 449, 450, 451, 452, 453, 412, 419, 399, 437, 429, 279, 423, 426, 427]
CHIN_IDX = [43, 106, 182, 83, 18, 313, 406, 335, 273, 422, 430, 394, 379, 378, 400, 377, 152, 148, 176, 149, 150, 169, 210, 202, 43]

HEMI_FACE_TOP_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 366, 137, 234, 127, 162, 21, 54, 103, 67, 109, 10]
HEMI_FACE_BOTTOM_IDX = [366, 401, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 177, 137, 366]
HEMI_FACE_LEFT_IDX = [152, 148, 176, 149, 150, 136, 172, 58, 177, 137, 234, 127, 162, 21, 54, 103, 67, 109, 10, 152]
HEMI_FACE_RIGHT_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 366, 401, 288, 397, 365, 379, 378, 400, 377, 152, 10]


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
        
    routes: list of tuple
        A list of tuples containing overlapping points, forming a path.
    """
    
    # Connvert the input list to a two-column dataframe
    landmark_dataframe = pd.DataFrame([(landmark_set[i], landmark_set[i+1]) for i in range(len(landmark_set) - 1)], columns=['p1', 'p2'])
    routes = []

    # Initialise the first two points
    p1 = landmark_dataframe.iloc[0]['p1']
    p2 = landmark_dataframe.iloc[0]['p2']

    for i in range(0, landmark_dataframe.shape[0]):
        obj = landmark_dataframe[landmark_dataframe['p1'] == p2]
        p1 = obj['p1'].values[0]
        p2 = obj['p2'].values[0]

        current_route = (p1, p2)
        routes.append(current_route)
    
    return routes

# Preconstructed face region paths for use with facial manipulation functions. Landmarks below are convex polygons
LEFT_EYE_PATH = create_path(LEFT_EYE_IDX)
LEFT_IRIS_PATH = create_path(LEFT_IRIS_IDX)
RIGHT_EYE_PATH = create_path(RIGHT_EYE_IDX)
RIGHT_IRIS_PATH = create_path(RIGHT_IRIS_IDX)
NOSE_PATH = create_path(NOSE_IDX)
LIPS_PATH = create_path(LIPS_IDX)
LIPS_TIGHT_PATH = create_path(LIPS_TIGHT_IDX)
FACE_OVAL_PATH = create_path(FACE_OVAL_IDX)
FACE_OVAL_TIGHT_PATH = create_path(FACE_OVAL_TIGHT_IDX)
HEMI_FACE_TOP_PATH = create_path(HEMI_FACE_TOP_IDX)
HEMI_FACE_BOTTOM_PATH = create_path(HEMI_FACE_BOTTOM_IDX)
HEMI_FACE_LEFT_PATH = create_path(HEMI_FACE_LEFT_IDX)
HEMI_FACE_RIGHT_PATH = create_path(HEMI_FACE_RIGHT_IDX)

# The following landmark regions need to be partially computed in place, but paths have been created so they can still be 
# passed to the facial manipulation family of functions. Landmarks below are concave polygons.
CHEEKS_PATH = [(0,)]
LEFT_CHEEK_PATH = [(1,)]
RIGHT_CHEEK_PATH = [(2,)]
CHEEKS_NOSE_PATH = [(3,)]
BOTH_EYES_PATH = [(4,)]
FACE_SKIN_PATH = [(5,)]
CHIN_PATH = [(6,)]

# Masking options for mask_face_region
FACE_OVAL_MASK = 1
FACE_SKIN_MASK = 2
EYES_MASK = 3
IRISES_MASK = 21
LIPS_MASK = 22
HEMI_FACE_LEFT_MASK = 23
HEMI_FACE_RIGHT_MASK = 24
HEMI_FACE_BOTTOM_MASK = 25
HEMI_FACE_TOP_MASK = 26
EYES_NOSE_MOUTH_MASK = 14
MASK_OPTIONS = [FACE_OVAL_MASK, FACE_SKIN_MASK, EYES_MASK, IRISES_MASK, LIPS_MASK, HEMI_FACE_LEFT_MASK,
                HEMI_FACE_RIGHT_MASK, HEMI_FACE_BOTTOM_MASK, HEMI_FACE_TOP_MASK, EYES_NOSE_MOUTH_MASK]

# Compatible color spaces for extract_color_channel_means and face_color_shift
COLOR_SPACE_RGB = cv.COLOR_BGR2RGB
COLOR_SPACE_HSV = cv.COLOR_BGR2HSV_FULL
COLOR_SPACE_GRAYSCALE = cv.COLOR_BGR2GRAY
COLOR_SPACES = [COLOR_SPACE_RGB, COLOR_SPACE_HSV, COLOR_SPACE_GRAYSCALE]

COLOR_RED = 4
COLOR_BLUE = 5
COLOR_GREEN = 6
COLOR_YELLOW = 7

# Coloring regions
REGION_FULL_FACE = 15
REGION_CHEEKS = 16
REGION_CHEEKS_NOSE = 17
COLOR_REGION_OPTIONS = [REGION_FULL_FACE, REGION_CHEEKS, REGION_CHEEKS_NOSE]

# Fill options for occluded face regions
OCCLUSION_FILL_BLACK = 8
OCCLUSION_FILL_MEAN = 9
OCCLUSION_FILL_BAR = 10

# Blurring methods
BLUR_METHOD_AVERAGE = 11
BLUR_METHOD_GAUSSIAN = 12
BLUR_METHOD_MEDIAN = 13

# Noise methods
NOISE_METHOD_PIXELATE = 18
NOISE_METHOD_SALT_AND_PEPPER = 19
NOISE_METHOD_GAUSSIAN = 20

# Shuffling methods
LOW_LEVEL_GRID_SCRAMBLE = 27
HIGH_LEVEL_GRID_SCRAMBLE = 28
LANDMARK_SCRAMBLE = 29

# Optical Flow types
SPARSE_OPTICAL_FLOW = 30
DENSE_OPTICAL_FLOW = 31

# Point Light Display History Modes
SHOW_HISTORY_ORIGIN = 32
SHOW_HISTORY_RELATIVE = 33

# Frame Shuffle Modes
SHUFFLE_FRAME_ORDER = 34
REVERSE_FRAME_ORDER = 35

def calculate_rot_angle(slope1:float, slope2:float = 0):
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

    Returns:
    --------

    A point (x,y) intersecting the provided line, or None if no such point exists.
    """
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
            sys.exit(1)
        
        size = (int(capture.get(3)), int(capture.get(4)))
        result = cv.VideoWriter(output_dir + "\\" + filename + "_transcoded.mp4",
                                cv.VideoWriter.fourcc(*'MP4V'), 30, size)
        if not result.isOpened():
            print("Transcode_video: Error opening VideoWriter object.")
            sys.exit(1)
        
        while True:
            success, frame = capture.read()
            if not success:
                break

            result.write(frame)

        capture.release()
        result.release()


# Defining useful timing functions
def constant(t:float, **kwargs) -> float:
    return 1.0

def sigmoid(t:float, **kwargs) -> float:
    k = 1
    if "k" in kwargs:
        k = kwargs["k"]

    return 1/(1 + np.exp(-k * t))

def linear(t:float, **kwargs) -> float:
    ''' Normalised linear timing function.

    Parameters
    ----------

    t: float
        The current time value of the video file being processed.
    
    kwargs: dict
        The linear timing function requires a start and end time, typically you will pass 0, and the video duration
        as start and end values. 
    
    Returns
    -------

    weight: float
    '''
    start = 0.0
    if "start" in kwargs:
        start = kwargs["start"]
    
    # end kwarg is always passed internally by package functions
    end = kwargs["end"]

    return (t-start) / (end-start)