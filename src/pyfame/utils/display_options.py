def display_convex_landmark_paths() -> None:
    print(
        "Predefined Convex Landmark Paths: \n"
        "   - LEFT_EYE_PATH\n"
        "   - LEFT_IRIS_PATH\n"
        "   - RIGHT_EYE_PATH\n"
        "   - RIGHT_IRIS_PATH\n"
        "   - NOSE_PATH\n"
        "   - MOUTH_PATH\n"
        "   - LIPS_PATH\n"
        "   - FACE_OVAL_PATH\n"
        "   - FACE_OVAL_TIGHT_PATH\n"
        "   - HEMI_FACE_TOP_PATH\n"
        "   - HEMI_FACE_BOTTOM_PATH\n"
        "   - HEMI_FACE_LEFT_PATH\n"
        "   - HEMI_FACE_RIGHT_PATH\n"
    )

def display_concave_landmark_paths() -> None:
    print(
        "Predefined Concave Landmark Paths: \n"
        "   - CHEEKS_PATH\n"
        "   - LEFT_CHEEK_PATH\n"
        "   - RIGHT_CHEEK_PATH\n"
        "   - CHEEKS_NOSE_PATH\n"
        "   - BOTH_EYES_PATH\n"
        "   - FACE_SKIN_PATH\n"
        "   - CHIN_PATH\n"
    )

def display_all_landmark_paths() -> None:
    print(
        "Predefined Landmark Paths: \n"
        "Convex paths:\n"
        "   - LEFT_EYE_PATH\n"
        "   - LEFT_IRIS_PATH\n"
        "   - RIGHT_EYE_PATH\n"
        "   - RIGHT_IRIS_PATH\n"
        "   - NOSE_PATH\n"
        "   - MOUTH_PATH\n"
        "   - LIPS_PATH\n"
        "   - FACE_OVAL_PATH\n"
        "   - FACE_OVAL_TIGHT_PATH\n"
        "   - HEMI_FACE_TOP_PATH\n"
        "   - HEMI_FACE_BOTTOM_PATH\n"
        "   - HEMI_FACE_LEFT_PATH\n"
        "   - HEMI_FACE_RIGHT_PATH\n"
        "Concave paths:\n"
        "   - CHEEKS_PATH\n"
        "   - LEFT_CHEEK_PATH\n"
        "   - RIGHT_CHEEK_PATH\n"
        "   - CHEEKS_NOSE_PATH\n"
        "   - BOTH_EYES_PATH\n"
        "   - FACE_SKIN_PATH\n"
        "   - CHIN_PATH\n"
    )

def display_face_mask_options() -> None:
    print(
        "Face Masking Method Options:\n"
        "   - FACE_OVAL_MASK (literal [1])\n"
        "   - FACE_SKIN_MASK (literal [2])\n"
        "   - EYES_MASK (literal [3])\n"
        "   - IRISES_MASK (literal [21])\n"
        "   - LIPS_MASK (literal [22])\n"
        "   - HEMI_FACE_LEFT_MASK (literal [23])\n"
        "   - HEMI_FACE_RIGHT_MASK (literal [24])\n"
        "   - HEMI_FACE_BOTTOM_MASK (literal [25])\n"
        "   - HEMI_FACE_TOP_MASK (literal [26])\n"
        "   - EYES_NOSE_MOUTH_MASK (literal [14])\n"
    )

def display_color_space_options() -> None:
    print(
        "Color Space Options:\n"
        "   - COLOR_SPACE_RGB\n"
        "   - COLOR_SPACE_HSV\n"
        "   - COLOR_SPACE_GRAYSCALE\n"
    )

def display_shift_color_options() -> None:
    print(
        "Face Color Shift Coloring Options:\n"
        "   - COLOR_RED (literal [4])\n"
        "   - COLOR_BLUE (literal [5])\n"
        "   - COLOR_GREEN (literal [6])\n"
        "   - COLOR_YELLOW (literal [7])\n"
    )

def display_occlusion_fill_options() -> None:
    print(
        "Occlusion Fill Options:\n"
        "   - OCCLUSION_FILL_BLACK (literal [8])\n"
        "   - OCCLUSION_FILL_MEAN (literal [9])\n"
        "   - OCCLUSION_FILL_BAR (literal [10])\n"
    )

def display_blur_method_options() -> None:
    print(
        "Blurring Method Options:\n"
        "   - BLUR_METHOD_AVERAGE (literal [11])\n"
        "   - BLUR_METHOD_GAUSSIAN (literal [12])\n"
        "   - BLUR_METHOD_MEDIAN (literal [13])\n"
    )

def display_noise_method_options() -> None:
    print(
        "Noise Method options:\n"
        "   - NOISE_METHOD_PIXELATE (literal [18])\n"
        "   - NOISE_METHOD_SALT_AND_PEPPER (literal [19])\n"
        "   - NOISE_METHOD_GAUSSIAN (literal [20])\n"
    )

def display_scramble_method_options() -> None:
    print(
        "Facial Scramble Method Options:\n"
        "   - LOW_LEVEL_GRID_SCRAMBLE (literal [27])\n"
        "   - HIGH_LEVEL_GRID_SCRAMBLE (literal [28])\n"
        "   - LANDMARK_SCRAMBLE (literal [29])\n"
    )

def display_optical_flow_options() -> None:
    print(
        "Optical Flow Parameter Options:\n"
        "   - SPARSE_OPTICAL_FLOW (literal [30])\n"
        "   - DENSE_OPTICAL_FLOW (literal [31])\n"
    )

def display_history_mode_options() -> None:
    print(
        "History Mode Options:\n"
        "   - SHOW_HISTORY_ORIGIN (literal [32])\n"
        "   - SHOW_HISTORY_RELATIVE (literal [33])\n"
    )

def display_shuffle_method_options() -> None:
    print(
        "Frame Shuffle Method Options:\n"
        "   - FRAME_SHUFFLE_RANDOM (literal [34])\n"
        "   - FRAME_SHUFFLE_RANDOM_W_REPLACEMENT (literal [35])\n"
        "   - FRAME_SHUFFLE_REVERSE (literal [36])\n"
        "   - FRAME_SHUFFLE_RIGHT_CYCLIC_SHIFT (literal [37])\n"
        "   - FRAME_SHUFFLE_LEFT_CYCLIC_SHIFT (literal [38])\n"
        "   - FRAME_SHUFFLE_PALINDROME (literal [39])\n"
        "   - FRAME_SHUFFLE_INTERLEAVE (literal [40])\n"
    )

def display_timing_function_options() -> None:
    print(
        "Available Timing Functions:\n"
        "   constant(t:float) -> always returns 1.0\n\n"
        "   linear(t:float, **kwargs) -> float in [0.0, 1.0]\n"
        "       **kwargs:   'start' - The start of the linear output range (default is 0.0)\n"
        "                   'end' - The end of the linear output range (default is vid_duration - 1.0)\n\n"
        "   sigmoid(t:float, **kwargs) -> float in [0.0, 1.0]\n"
        "       **kwargs:   'k' - The scaling factor of the sigmoid curve (default is 1.0)\n\n"
        "   gaussian(t:float, **kwargs) -> float in [0.0, 1.0]\n"
        "       **kwargs:   'mean' - The mean or center of the gaussian distribution (default is 0.0)\n"
        "                   'sigma' - The distribution or spread of the gaussian distribution (default is 1.0)\n\n"
    )