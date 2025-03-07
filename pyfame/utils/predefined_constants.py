import cv2 as cv

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

# Shift Color options
COLOR_RED = 4
COLOR_BLUE = 5
COLOR_GREEN = 6
COLOR_YELLOW = 7

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

# Facial Scrambling methods
LOW_LEVEL_GRID_SCRAMBLE = 27
HIGH_LEVEL_GRID_SCRAMBLE = 28
LANDMARK_SCRAMBLE = 29

# Optical Flow types
SPARSE_OPTICAL_FLOW = 30
DENSE_OPTICAL_FLOW = 31

# Point Light Display History Modes
SHOW_HISTORY_ORIGIN = 32
SHOW_HISTORY_RELATIVE = 33

# Frame Shuffle methods
FRAME_SHUFFLE_RANDOM = 34
FRAME_SHUFFLE_RANDOM_W_REPLACEMENT = 35
FRAME_SHUFFLE_REVERSE = 36
FRAME_SHUFFLE_RIGHT_CYCLIC_SHIFT = 37
FRAME_SHUFFLE_LEFT_CYCLIC_SHIFT = 38
FRAME_SHUFFLE_PALINDROME = 39
FRAME_SHUFFLE_INTERLEAVE = 40

SHUFFLE_METHODS = [FRAME_SHUFFLE_RANDOM, FRAME_SHUFFLE_RANDOM_W_REPLACEMENT, FRAME_SHUFFLE_REVERSE, 
                   FRAME_SHUFFLE_RIGHT_CYCLIC_SHIFT, FRAME_SHUFFLE_LEFT_CYCLIC_SHIFT, FRAME_SHUFFLE_PALINDROME,
                   FRAME_SHUFFLE_INTERLEAVE]