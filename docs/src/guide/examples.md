---
layout: doc
title: Examples
prev: false
next: false
---

## Getting Started {#start}

### Installation: {#install} 

#### General Users
PyFAME can be downloaded directly with pip via PyPi:

``` sh
pip install pyfame
```

#### Developers
For developers who are looking to contribute to the project, PyFAME can be cloned in typical SSH fashion:

``` sh
git clone git@github.com:Gavin-Bosman/PyFAME.git
```

Once you have the repository cloned locally, you will need to install dependencies via

``` sh
pip install -r requirements.txt
```

Additionally, if you would like to be able to run the test suite and documentation site, you will need to install additional dev dependencies:

::: info
This documentation site is built with vitepress, which requires you to have both Node.js>=20.0.0 and npm>=10.0.0 installed on your system. For more information on installing and setting up Node, [see here](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)
:::

``` sh
    pip install -r requirements_dev.txt
    npm install
```

## Using PyFAME {#using_pyfame}

The PyFAME package is divided into two main submodules `core` and `utils`. Simply importing pyfame will expose both submodules and the entirety of the package's contents. However, most users will be focused on using the core functionalities. So, generally PyFAME will be imported like this
``` python
import pyfame.core as pf
```

The `core` submodule contains several submodules itself, breaking up the package into specific functional groups, which is detailed below.

| Submodule Name | Functions Included | Description |
| :------------- | :----------------- | :---------- |
| Analysis       | get_optical_flow(), extract_face_color_means() | Provides various image and video analysis functions, including color and motion analysis of images and videos. |
| Coloring       | face_color_shift(), face_saturation_shift(), face_brightness_shift() | Contains a family of functions that all involve manipulating the color channels of images and video frames. All of the Coloring functions have access to timing functions to allow for temporal modulation of colour. |
| Exceptions     | None | This submodule contains various custom exception classes for the PyFAME package. |
| Occlusion      | mask_face_region(), occlude_face_region(), blur_face_region(), apply_noise() | Provides various functions commonly associated with obstructing or exposing the face in some way. The occlusion, blurring and noise functions all accept an image mask parameter to allow for dynamic regional occlusion. |
| Point_light_display | generate_point_light_display() | This submodule contains one function, which takes an input video file and a defined set of landmark points, and generates a point-light display of the landmarks provided. |
| Scrambling     | facial_scramble() | Provides two methods of performing facial scrambling: landmark-scrambling and grid-scrambling. |
| Temporal_transforms | generate_shuffled_block_array(), shuffle_frame_order() | Contains several functions involved with temporal (time-based) manipulations of video files. |


The `utils` submodule also contains several submodules, each providing various utility functions, predefined constants, and any extra features not directly relevant to the core funtionality of the package.

| Submodule Name | Description |
| :------------- | :---------- |
| Display_options | A group of functions that display parameter options to the terminal, i.e. `display_mask_type_options()`. |
| Landmarks      | Predefined landmark regions for use with all of the core functions. |
| Predefined_constants | Evident from this submodules name, it contains a large set of predefined parameter values for use with all of the core functions. |
| Setup_logging  | Provides access to a function `setup_logging()` which allows users to provide a custom logging config.yml if they want to define custom logging behaviour. |
| Timing_functions | A set of predefined timing functions, namely `constant()`, `linear()`, `sigmoid()` and `gaussian()`. |
| Utils     | Any extra utilities and mathematical operations not part of the core functionality. |

## Examples {#examples}

comming soon!