---
layout: doc
title: Api Overview
prev: false
next: 
    text: 'Moviefy'
    link: '/reference/moviefy'
---

# Overview

PyFAME: The Python Facial Analysis and Manipulation Environment is a Python toolkit for performing a variety of classical facial psychology manipulations over both still images and videos. All of PyFAME's manipulation functions can be layered, and they are designed in such a way that users can apply several manipulations in succession easily. Thus, PyFAME can be used to perform individual facial psychology experiments, or to create novel facial psychology stimuli which themselves can be used in experiments or as inputs to train neural networks.

The PyFAME package is divided into two main submodules `core` and `utils`. A top-level import of PyFAME will expose all of the `core` functions, and the `utils` submodule to the user. The package is set up this way so that users can simply import PyFAME like below in order to directly access all of the relevant functions.
``` python
import pyfame as pf

# pyfame.core functions are all exposed in the top-level import
pf.occlude_face_region(...)

# Extra utility functions like the parameter display functions are 
# available in pyfame.utils
pf.utils.display_all_landmark_paths()

# Specific commonly used utils submodules are exposed at the top level 
# This includes all predefined constants, landmarks and timing functions
pf.OCCLUSION_FILL_BAR
pf.HEMI_FACE_LEFT_PATH
pf.sigmoid()
```

Every facial manipulation function in PyFAME leverages MediaPipe's FaceMesh solution to identify and track facial landmarks. Thus, almost all of the functions take in some common parameters. Namely, `min_detection_confidence` and `min_tracking_confidence` are passed directly to the declaration of the MediaPipe FaceMesh model as follows.

``` python
import mediapipe as mp

face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, min_detection_confidence = min_detection_confidence,
                                    min_tracking_confidence = min_tracking_confidence, static_image_mode = static_image_mode)
```

For more information on MediaPipe's FaceMesh solution, [see here.](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

The `core` submodule contains several submodules itself, breaking up the package into specific functional groups, which is detailed below.

## Moviefy

The moviefy submodule contains some special helper functions for converting a directory of still images into "movies", by repeating and interpolating the images as video frames. This function makes it possible to perform PyFAME's spatial and temporal manipulations (i.e. color-shifting, frame-reordering) over static images. The two functions provided in the moviefy submodule are as follows

`normalize_image_sizes()` takes in an input directory path, and scans through the directory to find the maximum, and minimum image dimensions. Then, using the normalization method of choice (pad or crop), the function normalizes the image dimensions of all the images within the directory (in-place). This function can be used as a preprocessing step to `moviefy_images()`, or you can perform both steps in a single call by specifying `normalize=True` in the `moviefy_images()` call. 

`moviefy_images()` at the most basic level, will repeat the images provided to it a set number of times, written out as frames to an mp4 file. However, the function also offers an image interpolation option, which will blend successive images together over a transitionary period defined by input parameter `blended_frames_prop`. For example, with `repeat_duration = 1000`(msec), and with `blend_frames_prop = 0.25`, the last 250 msec of each 'block' of frames will be used to interpolate the first image into the next. 

## Analysis {#module_analysis}

This module provides several functions for extracting color and motion information from both image and video files.

`extract_face_color_means()` allows users to specify a sampling region of the face, a sampling frequency, and a color space to sample in. Only a single sample is performed over still images, while videos are sampled at periodic timestamps determined by the sampling frequency. Extracted color channel information is written out into a CSV file.

`get_optical_flow()` provides users the ability to perform both sparse (Lucas-Kanadae's algorithm) and dense (Farneback's algorithm) optical flow. This function provides two output files for each file input; a video file visualizing the optical flow, and a csv file containing periodic samples (determined by sampling frequency parameter) of the optical flow vectors.

## Coloring {#module_coloring}

This module contains a variety of functions for manipulating color in the facial region over still images and video files. 

`face_color_shift()` provides users the ability to manipulate and shift various color channels over the face region. This function operates in the BGR color space, and can manipulate the colors red, green, blue and yellow over the facial region. These color manipulations can also be modulated dynamically using a timing function. PyFAME contains several predefined timing functions, namely `linear()`, `gaussian()`, `sigmoid()` and `constant()`, but also allows users to define their own timing functions (at the risk of unexpected results).

`face_saturation_shift()` and `face_brightness_shift()` operate in a near-identical manner to `face_color_shift()`. They manipulate image or frame saturation (using the HSV color space) and brightness respectively. Additionally, both functions can be modulated with PyFAME's predefined timing functions, as seen with `face_color_shift()` above.

## Occlusion {#module_occlusion}

This module contains several functions associated with occluding, obstructing or cutting out regions of the face. Again, all of these functions operate over still images and videos. 

`mask_face_region()` provides users the ability to dynamically mask individual, or several facial regions at once. For specific use cases (i.e. green/blue screen) a background_color parameter is provided, with the default being white. Much of `mask_face_region()`'s functionality is utilised in almost every `Core` function, in order to allow dynamic regional application of the various manipulations. 

`occlude_face_region()` provides users access to several classical facial occlusion methods, including bubble occlusion, bar occlusion, and dynamic regional occlusion with custom fill colors. All occlusion types are set up to positionally lock onto, and track the face, making it simple to dynamically occlude the face in video files. 

`blur_face_region()` allows users to apply classical image and video blurring methods restricted over the facial region. This function can perform gaussian blurring, average blurring, and median blurring. 

`apply_noise()` functions similarly to `blur_face_region()`, but it encorporates more general noise methods and does not restrict the noise only to the facial region. This function can pixelate the face, apply gaussian noise, as well as salt and pepper noise. All of the noise methods are highly customizable, with input parameters such as `noise_prob`, `mean` and `standard_dev`.

## Point_light_display {#module_pld}

This module contains only one function, namely `generate_point_light_display()`. A point-light-display is a motion-perception paradigm that allows researchers to study how the brain perceives and interprets biological motion. This function focusses on the underlying face, overlaying pertinent landmarks with point-lights and tracking their position/velocity. Classically, point-light-displays have been created using motion capture software, which is both costly and requires physical labour. Alternatively, `generate_point_light_display()` is able to take any video containing a face, and overlay up to 468 unique points to generate a dynamic point-light-display.

One novel ability of `generate_point_light_display()` is it's ability to display historical displacement vectors. The function allows users to specify the history time window, as well as several methods of displaying the history vectors (relative positional history, relative to origin history).

## Scrambling {#module_scrambling}

Again, this module only contains one function, namely the `facial_scramble()` function. However, this function is multimodal and leverages several distinct methods of shuffling/scrambling the facial features. The two main scrambling methods provided are `landmark_scramble` and `grid_scramble`. These methods shuffle the facial features by masking out specified landmarks and randomizing their positions, and breaking up the face into a grid then repositioning the grid-squares respectively. 

## Temporal_transforms {#module_tt}

This module contains two related functions `generate_shuffled_block_array()` and `shuffle_frame_order()`. `shuffle_frame_order()` provides a variety of methods (i.e. palindrome, random sampling, cyclic shift) to temporally shift and restructure input video files. It performs the shuffling by breaking up the video frames into 'blocks' of frames, for which the time duration is specified by the user. `generate_shuffled_block_array()` is a helper function that returns a specific tuple which can be fed directly as input into `shuffle_frame_order()`. Depending on input parameters, `generate_shuffled_block_array()` returns a tuple containing the `block_order` array, the `block_size` and `block_duration`.

## Utils {#module_utils}

The `utils` submodule also contains several submodules, each providing various utility functions, predefined constants, and any extra features not directly relevant to the core funtionality of the package.

| Submodule Name | Description |
| :------------- | :---------- |
| Display_options | A group of functions that display parameter options to the terminal, i.e. `display_mask_type_options()`. |
| Landmarks      | Predefined landmark regions for use with all of the core functions. |
| Predefined_constants | Contains a large set of predefined parameter values for use with all of the core functions. |
| Setup_logging  | Provides access to a function `setup_logging()` which allows users to provide a custom logging config.yml if they want to define custom logging behaviour. |
| Timing_functions | A set of predefined timing functions, namely `constant()`, `linear()`, `sigmoid()` and `gaussian()`. |
| Utils     | Any extra utilities and mathematical operations not part of the core functionality. (i.e. `create_path()`, `compute_line_intersection()`) |
