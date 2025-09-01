---
layout: doc
title: Analyse
prev: 
    text: 'Overview'
    link: '/reference/overview'
next: 
    text: 'Coloring'
    link: '/reference/coloring'
---
# Analyse Module

## analyse_facial_colour_means()

```Python
analyse_facial_colour_means(
    file_paths:pandas.DataFrame,
    colour_space:int|str,
    min_detection_confidence:float,
    min_tracking_confidence:float
) -> dict[str, pandas.DataFrame]
```
Takes in any number of file paths, and for each file read-in, computes regional facial colour means in the specified colour_space. This function allows you to select a colour space from RGB, HSV or Greyscale. A mean sampling of the colour channels is performed over the cheeks, nose, chin and full-face, which is returned to the user as a dictionary of ("file name" : DataFrame) pairs.

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `file_paths` | `pandas.DataFrame` | An Nx2 dataframe of absolute and relative file paths, returned from the `make_paths()` function. |
| `colour_space` | `int` or `str` | Either a string literal ('rgb', 'hsv', etc.), or a predefined integer constant (i.e. COLOUR_SPACE_GREYSCALE), specifying the colour space of choice. |
| `min_detection_confidence` | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `min_tracking_confidence`  | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |

### Returns

`Dict[str : pandas.Dataframe]`

### Exceptions Raised
| Raises | Encountered Error |
| :----- | :---- |
| `OSError` | Given an invalid or broken file path-string. |
| `FileReadError` | If an error is encountered instantiating `cv2.VideoCapture()` or calling `cv2.imRead()`. |
| `UnrecognizedExtensionError` | If the function encounters an unrecognized image or video file extension. |

### Quick Example

```Python
import pyfame as pf

# Generate your file paths dataframe from the working directory
# which defaults to data/raw/ by default.
file_paths = pf.make_paths()

# Optionally filter out specific files
analyse_paths = file_paths.iloc([2:7], [0])     # Selects files 3-7, only absolute paths (first column)

# Perform analysis
colour_channel_analysis = pf.analyse_facial_colour_means(file_paths, 'hsv')

# Perform some operation over the analysis result,
# or write to disk
pf.analyse_to_disk(colour_channel_analysis)
```

## analyse_optical_flow_sparse()

```Python
analyse_optical_flow_sparse(
    file_paths:pandas.DataFrame,
    landmarks_to_track:list[int] | None,
    max_points:int,
    flow_accuracy_threshold:float, 
    output_detail_level:str, 
    frame_step:int,
    min_detection_confidence:float,
    min_tracking_confidence:float
) -> dict[str, pandas.DataFrame]
```

Takes in any number of file paths, and for each file compute and track the Lucas-Kanade sparse optical flow vectors. The vector magnitudes and angles are sampled at some interval of frames, determined by `frame_step`. Users may optionally pass a list of specific FaceMesh landmark id's, but by default up to `max_points` points will be automatically detected using the Shi-Tomasi corners algorithm. 

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `file_paths` | `pandas.DataFrame` | An Nx2 dataframe of absolute and relative file paths, returned from the `make_paths()` function. |
| `landmarks_to_track` | `list[int]` | An optional list of Mediapipe FaceMesh landmark id's. These points will be directly tracked or approximated with stronger points nearby. |
| `max_points` | `int` | The maximum number of points for the Shi-Tomasi corners algorithm to detect. |
| `flow_accuracy_threshold` | A termination criteria of the sparse optical flow calculation; defines the accuracy threshold at which the sparse flow calculation will stop iterating. |
| `output_detail_level` | `str` | One of "summary" or "full". Determines whether the vector sampling records aggregate statistics or full-depth statistics (sampling of every vector). |
| `frame_step` | `int` | The number of frames between successive optical flow calculations. Increase this value for more robust outputs. |
| `min_detection_confidence` | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `min_tracking_confidence`  | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |

### Returns

`dict[str, pandas.DataFrame]`

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
# Define your input and output file paths
in_dir = "c:/my/path/to/files"
out_dir = "c:/my/path/to/output"

# Simplest call
pf.extract_face_color_means(in_dir, out_dir, pf.COLOR_SPACE_RGB)

# Nested subdirectories in input folder, HSV output
pf.extract_face_color_means(in_dir, out_dir, pf.COLOR_SPACE_HSV, With_sub_dirs=True)
```