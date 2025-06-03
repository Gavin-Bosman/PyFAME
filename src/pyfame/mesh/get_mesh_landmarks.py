import pandas as pd

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

# pertinent facemesh landmark sets
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
MOUTH_IDX = [164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 186, 92, 165, 167, 164]
LIPS_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 0, 37, 39, 40, 185, 61]
LEFT_CHEEK_IDX = [207, 187, 147, 123, 116, 111, 31, 228, 229, 230, 231, 232, 233, 188, 196, 174, 217, 209, 49, 203, 206, 207]
RIGHT_CHEEK_IDX = [427, 411, 376, 352, 345, 340, 261, 448, 449, 450, 451, 452, 453, 412, 419, 399, 437, 429, 279, 423, 426, 427]
CHIN_IDX = [43, 106, 182, 83, 18, 313, 406, 335, 273, 422, 430, 394, 379, 378, 400, 377, 152, 148, 176, 149, 150, 169, 210, 202, 43]

HEMI_FACE_TOP_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 366, 137, 234, 127, 162, 21, 54, 103, 67, 109, 10]
HEMI_FACE_BOTTOM_IDX = [366, 401, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 177, 137, 366]
HEMI_FACE_LEFT_IDX = [152, 148, 176, 149, 150, 136, 172, 58, 177, 137, 234, 127, 162, 21, 54, 103, 67, 109, 10, 152]
HEMI_FACE_RIGHT_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 366, 401, 288, 397, 365, 379, 378, 400, 377, 152, 10]

# Preconstructed face region paths for use with facial manipulation functions. Landmarks below are convex polygons
LEFT_EYE_PATH = create_path(LEFT_EYE_IDX)
LEFT_IRIS_PATH = create_path(LEFT_IRIS_IDX)
RIGHT_EYE_PATH = create_path(RIGHT_EYE_IDX)
RIGHT_IRIS_PATH = create_path(RIGHT_IRIS_IDX)
NOSE_PATH = create_path(NOSE_IDX)
MOUTH_PATH = create_path(MOUTH_IDX)
LIPS_PATH = create_path(LIPS_IDX)
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