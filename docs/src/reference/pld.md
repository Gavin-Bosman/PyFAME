---
layout: doc
title: Point Light Display
prev: 
    text: 'Occlusion'
    link: '/reference/occlusion'
next: 
    text: 'Scrambling'
    link: '/reference/scrambling'
---
# Point-Light Display

## generate_point_light_display()

```python
generate_point_light_display(input_dir, output_dir, landmark_regions = [BOTH_EYES_PATH, MOUTH_PATH])
```
Creates a point-light display over the specified `landmark_regions`.

The `generate_point_light_display()` function provides users the ability to generate a facial point-light display with only an input video. Facial landmark regions of interest are passed in as paths (predefined, or created using `pyfame.utils.create_path()`). Furthermore, point density and color can be easily manipulated via input parameters. `generate_point_light_display()` also provides the ability to track and display point displacement history, which can be visualized in two methods. `SHOW_HISTORY_ORIGIN` visualizes the displacement vectors from the current point positions to their original positions, while `SHOW_HISTORY_RELATIVE` visualizes the displacement vectors as their relative path of travel within some time window defined by parameter `history_window_msec`.

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir` | `str` | A path string to the directory containing files to process. |
| `output_dir` | `str` | A path string to the directory where processed files will be output. |
| `landmark_regions` | `list[list[tuple]]` | A list of one or more facial landmark paths. These can be predefined or manually created using `pyfame.utils.create_path()`, which takes a list of landmark indicies and outputs a path. |
| `point_density` | `float` | A float in the range [0,1] that controls the spatial density of the points in the output point-light display. |
| `show_history` | `bool` | A boolean flag indicating whether or not to display the history vectors. |
| `history_mode` | `int` | An integer flag specifying the method of visualizing the history vectors; one of `pyfame.SHOW_HISTORY_ORIGIN` or `pyfame.SHOW_HISTORY_RELATIVE` |
| `history_window_msec` | `int` | The time duration in milliseconds that the relative history vectors will visualize. |
| `history_color` | `tuple[int]` | A BGR color code specifying the display color of the history vectors. |
| `point_color` | `tuple[int]` | A BGR color code specifying the display color of the points in the output point-light display. |
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

# Simplest call; defaults to full face PLD with no history vectors
pf.generate_point_light_display(in_dir, out_dir)

# PLD of the eyes, nose and mouth, with blue relative history vectors
pf.generate_point_light_display(
    in_dir, out_dir, landmark_regions = [pf.BOTH_EYES_PATH, pf.NOSE_PATH, pf.MOUTH_PATH],
    show_history = True, history_mode = pf.SHOW_HISTORY_RELATIVE, history_color = (255,0,0)
)
```