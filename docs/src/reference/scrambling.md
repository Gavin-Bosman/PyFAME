---
layout: doc
title: Point Light Display
prev: 
    text: 'Point-Light Display'
    link: '/reference/pld'
next: 
    text: 'Temporal Transforms'
    link: '/reference/temporal_transforms'
---
# Scrambling Module

## facial_scramble()

```python
facial_scramble(input_dir, output_dir, scramble_method = HIGH_LEVEL_GRID_SCRAMBLE)
```
Scrambles the positions of the facial features using the specified `scramble_method`.

The `facial_scramble()` function is essentially two functions disguised as one, providing users the ability to perform grid-based and landmark-based shuffling of the face. The input parameter `scramble_method` defines the type of facial scrambling to be performed, which can be `pyfame.HIGH_LEVEL_GRID_SCRAMBLE`, `pyfame.LOW_LEVEL_GRID_SCRAMBLE` or `pyfame.LANDMARK_SCRAMBLE`. High level grid scramble will reshuffle the facial grid squares purely randomly, while low level grid scramble will reshuffle the facial grid squares within their rows, with the bounds of their random positional reassignment determined by parameter `grid_scramble_threshold`. Landmark-based scrambling will cut out and store the eyes, eyebrows, nose, and mouth, and randomly reorient and reposition them over the face. As all of the scrambling methods heavily rely on random number generation, the random number generator can be seeded with parameter `rand_seed` to ensure reproducible results. 

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir` | `str` | A path string to the directory containing files to process. |
| `output_dir` | `str` | A path string to the directory where processed files will be output. |
| `out_grayscale` | `bool` | A boolean flag indicating if the output image or video should be written in grayscale. |
| `scramble_method` | `int` | An integer flag specifying the facial scrambling method. One of `pyfame.HIGH_LEVEL_GRID_SCRAMBLE`, `pyfame.LOW_LEVEL_GRID_SCRAMBLE` or `pyfame.LANDMARK_SCRAMBLE` |
| `rand_seed` | `int` | A seed to be passed to the random number generator, to ensure reproduceable results. |
| `grid_scramble_threshold` | `int` | An integer specifying the max horizontal distance an individual grid square can be randomly moved when performing a low-level grid scramble. |
| `grid_square_size` | `int` | An integer specifying the square dimensions of each individual grid square that makes up the overlayed facial grid. The default value of 40 defines (40px, 40px) grid squares. |
| `with_sub_dirs`            | `bool` | A boolean flag indicating if the input directory contains sub-directories. |
| `min_detection_confidence` | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `min_tracking_confidence`  | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |

### Returns

`None`

### Exceptions Raised

| Raises | Encountered Error |
| :----- | :---- |
| `ValueError` | Given unrecognized input parameter values. |
| `TypeError` | Given invalid input parameter typings. |
| `OSError` | Given an invalid path-string for either `input_dir` or `output_dir`. |
| `FileReadError` | If an error is encountered instantiating `cv2.VideoCapture()` or calling `cv2.imRead()`. |
| `FileWriteError` | If an error is encountered instantiating `cv2.VideoWriter()` or calling `cv2.imWrite()`. |
| `UnrecognizedExtensionError` | If the function encounters an unrecognized image or video file extension. |
| `FaceNotFoundError` | If the mediapipe FaceMesh model cannot identify a face in the input image or video. |

### Quick Example

```Python
import pyfame as pf

# Define input paths
in_dir = "c:/my/path/to/input/"
out_dir = "c:/my/path/to/output/"

# Simplest call; defaults to fully random grid-scramble, with grid square size of (40,40)
# It's always a good idea to define a random seed to ensure results are reproducable
pf.facial_scramble(in_dir, out_dir, rand_seed=1234)

# Landmark scramble in grayscale
pf.facial_scramble(
    in_dir, out_dir, rand_seed = 1234, 
    out_grayscale = True, scramble_method = pf.LANDMARK_SCRAMBLE
)

# Low level grid scrambling with grid square size of (30,30)
pf.facial_scramble(
    in_dir, out_dir, rand_seed = 1234,
    scramble_method = pf.LOW_LEVEL_GRID_SCRAMBLE, 
    grid_scramble_threshold = 2, grid_square_size = 30
)
```