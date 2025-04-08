---
layout: doc
title: Api Overview
prev: false
next: 
    text: 'Occlusion'
    link: '/reference/occlusion'
---

# Overview

<p>
    <img src="/pyfame_logo.png" width=400px style="float: center" />
</p>

PyFAME: The Python Facial Analysis and Manipulation Environment is a Python toolkit for performing a variety of classical facial psychology manipulations over both still images and videos. All of PyFAME's manipulation functions can be layered, and they are designed in such a way that users can apply several manipulations in succession easily. Thus, PyFAME can be used to perform individual facial psychology experiments, or to create novel facial psychology stimuli which themselves can be used in experiments or as inputs to train neural networks.

The PyFAME package is divided into two main submodules `core` and `utils`. Simply importing pyfame will expose both submodules and the entirety of the package's contents. However, most users will be focused on using the core functionalities. So, generally PyFAME will be imported as follows
``` python
import pyfame.core as pf
```

The `core` submodule contains several submodules itself, breaking up the package into specific functional groups, which is detailed below.

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

temp filler

## Scrambling {#module_scrambling}

temp filler

## Temporal_transforms {#module_tt}

temp filler


The `utils` submodule also contains several submodules, each providing various utility functions, predefined constants, and any extra features not directly relevant to the core funtionality of the package.

| Submodule Name | Description |
| :------------- | :---------- |
| Display_options | A group of functions that display parameter options to the terminal, i.e. `display_mask_type_options()`. |
| Landmarks      | Predefined landmark regions for use with all of the core functions. |
| Predefined_constants | Evident from this submodules name, it contains a large set of predefined parameter values for use with all of the core functions. |
| Setup_logging  | Provides access to a function `setup_logging()` which allows users to provide a custom logging config.yml if they want to define custom logging behaviour. |
| Timing_functions | A set of predefined timing functions, namely `constant()`, `linear()`, `sigmoid()` and `gaussian()`. |
| Utils     | Any extra utilities and mathematical operations not part of the core functionality. |

