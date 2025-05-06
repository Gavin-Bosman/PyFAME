---
layout: doc
title: Codebook
prev: 
    text: 'Utilities'
    link: '/reference/utils'
next: False
aside: False
---
# PyFAME's Codebook

Here you can view PyFAME's complete codebook, containing detailed descriptions and metadata associated with all variables defined within the package. 

| Variable Name | Description | Source File |
| ------------- | ----------- | ----------- |
| `FACE_OVAL_MASK` | A `mask_type` option for `pyfame.mask_face_region()`, specifying the standard full-face oval mask. | `predefined_constants.py` |
| `FACE_SKIN_MASK` | A `mask_type` option for `pyfame.mask_face_region()`, specifying the full-face oval minus the eyes and mouth. | `predefined_constants.py` |
| `EYES_MASK` | A `mask_type` option for `pyfame.mask_face_region()`, specifying a mask of both eyes and eyebrows. |  `predefined_constants.py` |
| `IRISES_MASK` | A `mask_type` option for `pyfame.mask_face_region()`, specifying a mask of both eyes not including the eyebrows. | `predefined_constants.py` |
| `LIPS_MASK` | A `mask_type` option for `pyfame.mask_face_region()`, specifying a mask around the lips/mouth. | `predefined_constants.py` |
| `HEMI_FACE_LEFT_MASK` | A `mask_type` option for `pyfame.mask_face_region()`, specifying a left-half hemi-face mask. | `predefined_constants.py` |
| `HEMI_FACE_RIGHT_MASK` | A `mask_type` option for `pyfame.mask_face_region()`, specifying a right-half hemi-face mask. | `predefined_constants.py` |
| `HEMI_FACE_TOP_MASK` | A `mask_type` option for `pyfame.mask_face_region()`, specifying a top-half hemi-face mask. | `predefined_constants.py` |
| `HEMI_FACE_BOTTOM_MASK` | A `mask_type` option for `pyfame.mask_face_region()`, specifying a bottom-half hemi-face mask. | `predefined_constants.py` |
| `EYES_NOSE_MOUTH_MASK` | A `mask_type` option for `pyfame.mask_face_region()`, specifying a mask of the eyes, eyebrows, nose and mouth. | `predefined_constants.py` |
| `COLOR_RED` | A `shift_color` option for `pyfame.face_color_shift()`, specifying the color red. | `predefined_constants.py` |
| `COLOR_BLUE` | A `shift_color` option for `pyfame.face_color_shift()`, specifying the color blue. | `predefined_constants.py` |
| `COLOR_GREEN` | A `shift_color` option for `pyfame.face_color_shift()`, specifying the color green. | `predefined_constants.py` |
| `COLOR_YELLOW` | A `shift_color` option for `pyfame.face_color_shift()`, specifying the color yellow. | `predefined_constants.py` |
| `COLOR_SPACE_BGR` | A `color_space` option for `pyfame.extract_face_color_means`, specifying the OpenCV default BGR color space. | `predefined_constants.py` |
| `COLOR_SPACE_HSV` | A `color_space` option for `pyfame.extract_face_color_means`, specifying the HSV color space. | `predefined_constants.py` |
| `COLOR_SPACE_GRAYSCALE` | A `color_space` option for `pyfame.extract_face_color_means`, specifying the grayscale color space. | `predefined_constants.py` |
| `OCCLUSION_FILL_BLACK` | An `occlusion_fill` option for `pyfame.occlude_face_region()`, specifying for the occluded regions to be filled with black (0,0,0). | `predefined_constants.py` |
| `OCCLUSION_FILL_MEAN` | An `occlusion_fill` option for `pyfame.occlude_face_region()`, specifying for the occluded regions to be filled with the facial-skin's mean color. | `predefined_constants.py` |
| `OCCLUSION_FILL_BAR` | An `occlusion_fill` option for `pyfame.occlude_face_region()`, specifying for the occluded regions to be occluded by a horizontal bar. | `predefined_constants.py` |
| `BLUR_METHOD_AVERAGE` | A `blur_method` option for `pyfame.blur_face_region()`, specifying the average blurring method (`cv2.blur()`). | `predefined_constants.py` |
| `BLUR_METHOD_GAUSSIAN` | A `blur_method` option for `pyfame.blur_face_region()`, specifying the gaussian blurring method (`cv2.GaussianBlur()`). | `predefined_constants.py` |
| `BLUR_METHOD_MEDIAN` | A `blur_method` option for `pyfame.blur_face_region()`, specifying the median blurring method (`cv2.medianBlur()`). | `predefined_constants.py` |
| `NOISE_METHOD_PIXELATE` | A `noise_method` option for `pyfame.apply_noise()`, specifying for pixelation to be applied. | `predefined_constants.py` |
| `NOISE_METHOD_SALT_AND_PEPPER` | A `noise_method` option for `pyfame.apply_noise()`, specifying for salt and pepper noise to be applied. | `predefined_constants.py` |
| `NOISE_METHOD_GAUSSIAN` | A `noise_method` option for `pyfame.apply_noise()`, specifying for Gaussian noise to be applied. | `predefined_constants.py` |
| `LOW_LEVEL_GRID_SCRAMBLE` | A `scramble_method` option for `pyfame.facial_scramble()`, specifying a grid-based scrambling of the face with semi-random horizontal shuffling of the grid squares. This scrambling method allows you to shuffle grid squares of the face while maintaining vertical positioning of facial landmarks. | `predefined_constants.py` |
| `HIGH_LEVEL_GRID_SCRAMBLE` | A `scramble_method` option for `pyfame.facial_scramble()`, specifying a grid-based scrambling of the face with fully-random shuffling of the grid squares. | `predefined_constants.py` |
| `LANDMARK_SCRAMBLE` | A `scramble_method` option for `pyfame.facial_scramble()`, specifying landmark positional scrambling on the face. The eyes and brows, nose, and mouth are cut out of the face and filled with the mean skin tone. Each landmark is then randomly reoriented and positioned over the underlying face. | `predefined_constants.py` |
| `SPARSE_OPTICAL_FLOW` | An `optical_flow_type` option for `pyfame.get_optical_flow()`, specifying the use of the Lucas-Kanadae's sparse optical flow algorithm. | `predefined_constants.py` |
| `DENSE_OPTICAL_FLOW` | An `optical_flow_type` option for `pyfame.get_optical_flow()`, specifying the use of Farneback's dense optical flow algorithm. | `predefined_constants.py` |
| `SHOW_HISTORY_ORIGIN` | A `history_mode` option for `pyfame.generate_point_light_display()`, specifying that the history vectors should highlight each point's displacement from their origin. | `predefined_constants.py` |
| `SHOW_HISTORY_RELATIVE` | A `history_mode` option for `pyfame.generate_point_light_display()`, specifying that the history vectors should highlight each point's relative displacement history in the specified time window. | `predefined_constants.py` |
| `FRAME_SHUFFLE_RANDOM` | A `shuffle_method` option for `pyfame.shuffle_frame_order()` and `pyfame.generate_shuffled_block_array()`, specifying a fully-random shuffling of the input video's frame ordering. | `predefined_constants.py` |
| `FRAME_SHUFFLE_RANDOM_W_REPLACEMENT` | A `shuffle_method` option for `pyfame.shuffle_frame_order()` and `pyfame.generate_shuffled_block_array()`, specifying a random sampling with replacement of the input video's frame ordering. | `predefined_constants.py` |
| `FRAME_SHUFFLE_REVERSE` | A `shuffle_method` option for `pyfame.shuffle_frame_order()` and `pyfame.generate_shuffled_block_array()`, specifying a reversal of the input video's frame ordering. | `predefined_constants.py` |
| `FRAME_SHUFFLE_RIGHT_CYCLIC_SHIFT` | A `shuffle_method` option for `pyfame.shuffle_frame_order()` and `pyfame.generate_shuffled_block_array()`, specifying a right-cyclic rotation of the frame ordering within each block of frames. | `predefined_constants.py` |
| `FRAME_SHUFFLE_LEFT_CYCLIC_SHIFT` | A `shuffle_method` option for `pyfame.shuffle_frame_order()` and `pyfame.generate_shuffled_block_array()`, specifying a left-cyclic rotation of the frame ordering within each block of frames. | `predefined_constants.py` |
| `FRAME_SHUFFLE_PALINDROME` | A `shuffle_method` option for `pyfame.shuffle_frame_order()` and `pyfame.generate_shuffled_block_array()`, specifying that for each block of frames, the block is reversed and then appended to the video adjacent to the original block. This results in each temporal segment of the video to play the same forwards and backwards, similar to how textual palindrome words are read. | `predefined_constants.py` |
| `FRAME_SHUFFLE_INTERLEAVE` | A `shuffle_method` option for `pyfame.shuffle_frame_order()` and `pyfame.generate_shuffled_block_array()`, specifying for each block of frames to be interleaved with one another. | `predefined_constants.py` |
| `NORMALIZE_IMAGES_PAD` | A `normalization_method` option for `pyfame.moviefy_images()`, specifying for input image sizes to be normalized by padding each image to match the dimensions of the maximum image size provided. | `predefined_constants.py` |
| `NORMALIZE_IMAGES_CROP` | A `normalization_method` option for `pyfame.moviefy_images()`, specifying for input image sizes to be normalized by cropping each image to match the dimensions of the minimum image size provided. | `predefined_constants.py` |
| `FACE_OVAL_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate the face oval. | `landmarks.py` |
| `FACE_OVAL_TIGHT_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate a tighter face oval. This landmark set is provided as an alternative to `FACE_OVAL_IDX` for use with the `pyfame.coloring` family of functions. These landmarks are just inside the facial boundary, ensuring no background artifacts are included in the coloring manipulations. | `landmarks.py` | 
| `LEFT_EYE_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate the left eye and brow. | `landmarks.py` |
| `RIGHT_EYE_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate the right eye and brow. | `landmarks.py` |
| `LEFT_IRIS_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate the left iris. | `landmarks.py` |
| `RIGHT_IRIS_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate the right iris. | `landmarks.py` |
| `NOSE_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate the nose. | `landmarks.py` |
| `MOUTH_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate the mouth region. | `landmarks.py` |
| `LIPS_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate specifically the upper and lower lips. | `landmarks.py` |
| `LEFT_CHEEK_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate the left cheek. | `landmarks.py` |
| `RIGHT_CHEEK_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate the right cheek. | `landmarks.py` |
| `CHIN_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate the chin. | `landmarks.py` |
| `HEMI_FACE_LEFT_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate the left half hemi-face. | `landmarks.py` |
| `HEMI_FACE_RIGHT_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate the right half hemi-face. | `landmarks.py` |
| `HEMI_FACE_TOP_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate the top half hemi-face. | `landmarks.py` |
| `HEMI_FACE_BOTTOM_IDX` | A list containing the MediaPipe FaceMesh landmark indicies that encapsulate the bottom half hemi-face. | `landmarks.py` |
| `FACE_OVAL_PATH` | A closed landmark path specifying the face oval as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `FACE_OVAL_TIGHT_PATH` | A closed landmark path specifying the tight face oval as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `LEFT_EYE_PATH` | A closed landmark path specifying the left eye as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `RIGHT_EYE_PATH` | A closed landmark path specifying the right eye as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `LEFT_IRIS_PATH` | A closed landmark path specifying the left iris as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `RIGHT_IRIS_PATH` | A closed landmark path specifying the right iris as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `NOSE_PATH` | A closed landmark path specifying the nose as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `MOUTH_PATH` | A closed landmark path specifying the mouth as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `LIPS_PATH` | A closed landmark path specifying the lips as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `HEMI_FACE_LEFT_PATH` | A closed landmark path specifying the left half hemi-face as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `HEMI_FACE_RIGHT_PATH` | A closed landmark path specifying the right half hemi-face as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `HEMI_FACE_TOP_PATH` | A closed landmark path specifying the top half hemi-face as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `HEMI_FACE_BOTTOM_PATH` | A closed landmark path specifying the bottom half hemi-face as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `CHEEKS_PATH` | A concave polygon path, required to be computed in place. Each concave path is initialized to a unique placeholder, which is used to identify the specific path and help compute its indicies internally. This path specifies both cheeks as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `LEFT_CHEEK_PATH` | A concave polygon path, required to be computed in place. Each concave path is initialized to a unique placeholder, which is used to identify the specific path and help compute its indicies internally. This path specifies the left cheek as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `RIGHT_CHEEK_PATH` | A concave polygon path, required to be computed in place. Each concave path is initialized to a unique placeholder, which is used to identify the specific path and help compute its indicies internally. This path specifies the right cheek as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `CHEEKS_NOSE_PATH` | A concave polygon path, required to be computed in place. Each concave path is initialized to a unique placeholder, which is used to identify the specific path and help compute its indicies internally. This path specifies both cheeks and the nose as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `BOTH_EYES_PATH` | A concave polygon path, required to be computed in place. Each concave path is initialized to a unique placeholder, which is used to identify the specific path and help compute its indicies internally. This path specifies both eyes as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `FACE_SKIN_PATH` | A concave polygon path, required to be computed in place. Each concave path is initialized to a unique placeholder, which is used to identify the specific path and help compute its indicies internally. This path specifies the facial skin (the face oval minus the eyes, brows and lips) as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |
| `CHIN_PATH` | A concave polygon path, required to be computed in place. Each concave path is initialized to a unique placeholder, which is used to identify the specific path and help compute its indicies internally. This path specifies the chin as a region of interest. This path can be passed to almost all of the `pyfame.core` manipulations in order to define the region of application. | `landmarks.py` |

