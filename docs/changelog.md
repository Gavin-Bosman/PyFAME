# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- v0.7... unit testing
- v0.8... full docsite reimplementation with vitepress
- v1.0 gui preview prior to file processing

## [0.7.0] 2025-02-16

### Added

- New function `generate_shuffled_block_array()` has been implemented. This function was designed to be used alongside `shuffle_frame_order()` in order to abstract the block ordering from the user. `generate_shuffled_block_array()` takes in a file_path, shuffle_method and block_duration (in milliseconds) and returns a tuple of (block_order, block_size). This output can now directly be fed as an input parameter to `shuffle_frame_order()`, and `shuffle_frame_order()` will now call `generate_shuffled_block_array()` internally if no block_order is provided.
- Several new frame shuffling methods have been added including random sampling with replacement, left and right cyclic shift, palindrome shuffling and frame interleaving. 
- All functions now make use of logging. Standard logs can be found at PyFAME/logs/app.log. This file will be cycled once it reaches 5mb, and the last 5 log files will be stored in the logs folder. Error logs can be found at PyFAME/logs/error.log. This file will be rotated in a similar manner to the standard log. All functions now log their input parameters, various status updates on execution, and all parameter or execution errors that are raised.
- Logging configurations can be found in the PyFAME/config folder in `log_config.yaml`

### Changed

- Several internal changes to `shuffle_frame_order()`. Many of the array-randomizing functionalities have been rescoped to 
`generate_shuffled_block_array()`.
- Shuffle_methods have been given a new naming scheme, all of the constant variable names follow the following format: FRAME_SHUFFLE_{method name}.
- Bug fixes to `get_optical_flow()`, specifically the dense optical flow implementation a bug fix was required over the writing of flow vectors to a csv file.
- `Extract_face_color_means()` can now handle static images, on top of video files.

### Removed

## [0.6.7] - 2025-01-18

### Added

- Function `shuffle_frame_order()` can now take an input parameter `block_order`, a list of integers. When `block_order` is provided, `block_size` will be automatically computed based on the length of the block order list. For example, given a block order of [1,0,3,2,4,5], the function will compute `block_size` by taking the total frame count and dividing it by the length of the list.

### Changed

- Function `get_optical_flow()` now takes a parameter `optical_flow_type`. The function now includes the option to compute Farneback's dense optical flow on top of Lucas-Kanade sparse optical flow. Several other Farneback control parameters have been added to the function, but have all been defined with defaults and typical values in the function documentation.
- Furthermore, due to the sheer size of outputted csv files from `get_optical_flow()`, a new parameter `csv_sample_freq` has been added. This value (given in milliseconds) determines how frequently the function will write out optical flow vectors to the outputted csv file. This is implemented using a rolling time window, and comparing the current video timestamp with its value at each iteration of the running loop.

### Removed

## [0.6.6] - 2025-01-11

### Added

- New function `shuffle_frame_order()` has been implemented. The function has two running modes: `SHUFFLE_FRAME_ORDER` and `REVERSE_FRAME_ORDER`. Frames are read from the input video and stored in blocks, who's size is determined by input parameter `block_size`. Given an input video running at 30 fps, the default `block_size` of 30 will shuffle the order of roughly 1 second segments of the video file. The output order of the frame blocks is determined randomly. Users may pass a `rand_seed` to seed the rng for reproducable results.

### Changed

### Removed

## [0.6.5] - 2025-01-02

### Added

- Function `point_light_display()` has been further expanded. Now the function can display point displacement history, which can be drawn on the output file in one of two methods. Displacement history can be toggled on and off using the input parameter `show_history`. `SHOW_HISTORY_ORIGIN` will draw the displacement vector for each point relative to their original positions in frame 1. `SHOW_HISTORY_RELATIVE` will draw each points path history, displaying each path segment for a set amount of time given by input parameter `history_window_msec`.

### Changed

- Previously, removing points to satisfy the `point_density` was done purely randomly. However this caused issues abstracting the shapes of the landmark regions, making them difficult to identify. Point removal has been reimplemented using a normal gaussian distribution to control which points are removed and which are retained. As a result, even at lower densities the points stay clustered as to maintain the shape of the landmark they are tracking.
- Point size and color were previously hard-coded values. New parameters `point_color`, `point_radius`, and `history_color` allow the user to customize how the functions outputs appear.

### Removed

## [0.6.4] - 2024-12-30

### Added

- New function `point_light_display()` has been implemented. The function utilises the 478 landmark points tracked by the mediapipe FaceMesh to generate a point-light display of the face. The default functionality displays all 478 landmark points. However, users may provide landmark sets or predefined landmark sets from `pyfameutils` to the input parameter `landmarks_regions` to further specify which regions or landmarks will be included in the functions output. 
- Users may wish to manipulate the number of points without affecting the overall shape of the landmark regions. In order to do so, users may provide a floating point density to the input parameter `point_density`.

### Changed

### Removed

## [0.6.3] - 2024-12-22

### Added

- Expanded functionality for function `get_optical_flow()`. Now beyond outputing the visualised optical flow vectors, the function will output a csv containing the timestamp, previous and current (x,y) positions, vector magnitude, vector angle, status and error rate for each tracked point at every frame. 

### Changed

- `get_optical_flow()` has now been fully parameterized. The Lucas-Kanade optical flow control parameters were previously hard-coded values, but are now available to be passed as input parameters with predefined defaults. New parameters include `max_corners`, `corner_quality_lvl`, `min_corner_distance`, `win_size`, `max_pyr_lvl`, `max_lk_iter` and `lk_accuracy_thresh`.
- Both `point_color` and `vector_color` are now available as input parameters.

### Removed

## [0.6.2] - 2024-12-17

### Added

- New function `get_optical_flow()` has been implemented. The function makes use of the Shi-Tomasi corners algorithm to find good points to track, and passes these points on to the Lucas-Kanade sparse optical flow algorithm. At each new frame, the function draws a point over the current positions of the good points list, then using inter-frame movement draws motion vector histories of each point overtop of the input file. 
- Alternatively to using the Shi-Tomasi good corners algorithm, the user may provide a set of FaceMesh landmark id's to track within the input parameter `landmarks_to_track`.

### Changed

### Removed

## [0.6.1] - 2024-12-09

### Added

### Changed

- Bug causing grid-scramble order to be recomputed midway through file processing has been fixed.
- Bug causing jittery movement of facial landmarks when using `LANDMARK_SCRAMBLE` with video files has been fixed.

### Removed

## [0.6.0] - 2024-12-07

### Added

- Function `facial_scramble()` now takes an input parameter `scramble_method`. Three new scrambling methods have been defined within `pyfameutils`. These include `LOW_LEVEL_GRID_SCRAMBLE`, `HIGH_LEVEL_GRID_SCRAMBLE` and `LANDMARK_SCRAMBLE`.
- Landmark based scrambling is a new addition to the function. It takes the eyes, eyebrows, nose and mouth, and randomly swaps their positions and orientation. In order to provide seamless gaps between the underlying face and the manipulated landmark regions, the landmarks are cut out and stored, then the holes in the image are filled using the Telea image inpainting algorithm. Telea inpainting fills the image holes with a smooth gradient sampled from nearest neighbouring pixels.
- In order to seamlessly paste the facial landmarks back onto the face, cv2's SeamlessClone() method was used to ensure proper blending of landmark edges into the facial skin tone. 

### Changed

- Grid-based scrambling has been subdivided into `LOW_LEVEL_GRID_SCRAMBLE` and `HIGH_LEVEL_GRID_SCRAMBLE`. High level grid scrambling will function the same as previously implemented, with fully random shuffling of grid squares. Low level grid scrambling will make use of a new parameter `GRID_SCRAMBLE_THRESHOLD`; which defines the max x or y distance that a particular grid square can be moved from its original position.
- In order to reduce computation time from recomputing grid-square positions at each frame, the shuffled grid order is now precomputed prior to the functions main running loop. Beyond the grid-shuffle order, both grid-shuffling methods are implemented essentially the same. So precomputing the shuffle order allows for the removal of many lines of duplicate code.
- Many papers performing these types of facial scrambling/shuffling operate over grayscale images and video. As such `facial_scramble` now includes input parameter `out_grayscale` in order to toggle grayscale and color outputs.

### Removed

## [0.5.9] - 2024-12-02

### Added

- New function `facial_scramble` has been implemented. This function currently can perform grid-based shuffling of the face. Input parameter `grid_square_size` specifies the square dimensions (in pixels) of each grid square, and is used to compute the optimal grid arrangement to encapsulate the entire face with minimal background inclusions. 
- The grid shuffle order is computed randomly using NumPy's `default_rng()`. A random seed may be passed as an input parameter to ensure reproducable outputs. 

### Changed

### Removed

## [0.5.8] - 2024-11-24

### Added

- Bug fixes for `apply_noise()`.
- `apply_noise()` has been further expanded to include masking capabilities, and is now compatible with all predefined mask variables available in `pyfameutils`.

### Changed

### Removed

- Removed `random` as a dependency; all random operations are now computed using NumPy's random generator class.

## [0.5.7] - 2024-11-20

### Added

- New function `apply_noise()` has been implemented. This function provides three noise operations to select from; 
'pixelate', 'gaussian' and 'salt and pepper'. 
- `apply_noise()` provides a variety of customization options such as specifying the noise probability, mean and standard
deviation of the gaussian curve to be sampled from, as well as a random seed to be passed to the numpy random number generator. 
- An expanded set of masking options has been added to pyfameutils.MASK_OPTIONS for use with `mask_face_region()`. `mask_face_region` now also allows the user to specify the background color of output files via a BGR integer color code. 

### Changed

- Previous masking options FACE_OVAL and FACE_OVAL_TIGHT have been removed and replaced by the singular FACE_OVAL_MASK in hopes to alleviate any user confusion between the two options previously. 
- PsyFace has been officially renamed to PyFAME: the Python Facial Analysis and Manipulation Environment.

### Removed

## [0.5.6] - 2024-11-12

### Added

- Bug fixes for last major feature update (v0.5.5).
- `CHIN_PATH` has been added as a predefined path for use with all facial manipulation functions.
- `extract_color_channel_means` has been renamed as `extract_face_color_means`. The function now will not only output full-facial means, but also regional color means in the cheek, nose and chin areas for all colour spaces. 

### Changed

- The Hemi-face family of landmark paths have been converted to standard paths, and no longer require in-place computation. 
- `occlude_face_region`'s implementation of bar-style occlusion has been reworked, such that now the occluding bar will track correctly with the position of the face and axis of the head (the occluding bar no longer remains paralell to the horizontal axis).
- `face_color_shift`, `face_saturation_shift` and `face_brightness_shift` now only take list[list[tuple]] for input parameter `landmark_regions`. This massively reduces the ammount of duplicate code previously divided among if-else statements based on what was passed to `landmark_regions`.

### Removed

## [0.5.3 - 0.5.5] - 2024-10-16

### Added

- Major feature updates to all facial manipulation functions. `face_color_shift`, `face_saturation_shift`, `face_brightness_shift` and `blur_face_region` now are all compatible with timing functions, and every predefined landmark region defined within `psyfaceutils.py`. 
- Some of the landmark paths have been redefined as placeholders, as they either need to be calculated in place (hemi-face regions) or require a different method to draw the landmark polygons (Cheek landmark regions form concave polygons).
- `FACE_SKIN_PATH` constant has been defined in order to provided easier access to facial skin colouring, leaving the lips and eyes untouched. For similar ease of use reasons, other commonly used grouped regions have been defined, including `CHEEKS_PATH` and `CHEEKS_NOSE_PATH` for use in facial "blushing".

### Changed

### Removed

- Bad practice global variable declarations have been removed entirely. 

## [0.5.2] - 2024-10-09

### Added

- `Occlude_face_region` can now perform vertical and horizontal hemi-face occlusion. Hemi-face masks rely on the facial screen coords, thus they cannot be precomputed. However, predefined placeholder constants `HEMI_FACE_TOP`, `HEMI_FACE_BOTTOM`, `HEMI_FACE_LEFT`, and `HEMI_FACE_RIGHT` have been defined and can still be passed in `landmarks_to_occlude` as any of the other predefined landmark paths can. 
- Helper function `compute_line_intersection` has been created and can be found in `psyfaceutils.py`.
- Additional masking option `EYES_NOSE_MOUTH_MASK` has been added to `mask_face_region`.

### Changed

### Removed

- Predefined landmark paths `UPPER_FACE_PATH` and `LOWER_FACE_PATH` have been removed, and replaced with `HEMI_FACE_TOP`, and `HEMI_FACE_BOTTOM` respectively.

## [0.5.1] - 2024-10-03

### Added

- `Blur_face_region` provides dynamic facial blurring functionality with several blurring methods (average, gaussian, median) over user-specified facial regions. 
- Added horizontal hemi-face occlusion, `UPPER_FACE_PATH` and `LOWER_FACE_PATH` constants can be found in psyfaceutils.py.

### Changed

- `Face_luminance_shift` has been replaced with `face_brightness_shift`. `Face_brightness_shift` will now take an integer shift value in the range [-255, 255], with -255 and 255 representing pure black and white respectively. 

### Removed

- `Face_luminance_shift` has been removed due to buggy behaviour when manipulating image luminance.

## [0.5.0] - 2024-09-24

### Added

- Package documentation is now built with MKDocs
- `Face_saturation_shift` and `Face_luminance_shift` are now standalone functions, where previously saturation and luma parameters were passed to the Face_color_shift function. 
- Github.io hosting for documentation page, as well as refactored github landing page and readme.md.
- License.txt added to root project structure.

### Changed

- `Shift_color_temp` was refactored to be a nested function within `Face_color_shift`, saturation and luminance shifting were relocated to their own specific functions. 
- Floodfilling operation involved with foreground-background seperation had some buggy behaviour if there was any discontinuity in the background. An intermediate step was added where prior to floodfilling, the thresholded image is padded with a 10 pixel border, which is removed after the floodfill. This border ensures background continuity when performing the floodfill operation.
- Parameters `max_color_shift` and `max_sat_shift` are now renamed to `shift_magnitude`.

### Removed

- Sphinx and readthedocs project files and dependencies

## [0.4.2] - 2024-08-24

### Added

- Sphinx dependency for autodocumentation.
- Rst files defining the documentation build. 

### Changed

- Updated readme.md with examples, licenses and link to documentation page

### Removed

## [0.4.1] - 2024-08-18

### Added

### Changed

- v0.4.1 bug fixes for processing directories of mixed file types (images and videos). 

### Removed

## [0.4.0] - 2024-08-17

### Added

### Changed

- v0.4 Refactored all methods; moved repetative frame operations to nested functions for increased readability.
- Fixed buggy behaviour when working with still images over all methods. On top of video formats .MP4 and .MOV, you can now perform facial masking, occlusion and colour shifting over image formats .jpg, .jpeg, .png, and .bmp.
- Increased error handling; methods should now be able to process large directories of mixed file formats efficiently in a single call. 

### Removed

## [0.3.1] - 2024-08-11

### Added

- v0.3.1 Support for nose masking and occluding

### Changed

- Added bar-style occlusion options to occlude_face_region(). You can now perform bar-style occlusion on the eyes, nose 
and mouth regions. 

### Removed

## [0.3.0] - 2024-08-02

### Added

- v0.3 occlude_face_region()

### Changed

- Redefined the naming convention used for constants in utils.py

### Removed

## [0.2.2] - 2024-07-31

### Added

### Changed

- Changed mp4 video codec from h264 to cv2 supported mp4v.
- Mask_face_region and face_color_shift now take confidence parameters for the underlying mediapipe face landmarker model.
- Implemented otsu thresholding to isolate foreground to use as a mask. This foreground mask ensures that no background 
artifacts are present in the facial color shifting, or facial masking. 
- Added documentation for new function parameters.

### Removed

## [0.2.1] - 2024-07-24

### Added

- v0.2.1 transcode_video_to_mp4()

### Changed

- All functions will work by default with .mp4 and .mov video files. If an older container is being used, 
see transcode_video_to_mp4 to convert video codecs.
- Bug fixes with facial mask in face_color_shift; removed background artifacts present in the masked facial region.

### Removed
- Removed dependancy ffprobe-python.

## [0.2.0] - 2024-07-21

### Added

- v0.2 added dependancy ffprobe-python.

### Changed

- Added input file codec sniffing, output video files will now match input type for mask_face_region
and face_color_shift.

### Removed

## [0.1.1] - 2024-07-20

### Added

### Changed

- Minor bug fix for negative saturation shift.

### Removed

## [0.1.0] - 2024-07-17

### Added

- v0.1 mask_face_region()
- v0.1 extract_color_channel_means()
- v0.1 face_color_shift()
- v0.1 shift_color_temp()

### Changed

- Updated documentation and type hints for all package functions.
- Vectorized color shifting operations in shift_color_temp, massively reducing time costs.
- Restructured package into src, data and testing folders.
- Moved constants and helper functions into utils.py.

### Removed
