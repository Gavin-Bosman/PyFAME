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

```
For each image or video provided in `input_dir`, the facial color means in the provided color space are extracted and written out as a CSV.

This function will extract both the global (full-face) and local (landmark-specific) color channel values in the specified color space (determined by input parameter `color_space`). The color channel values of the full face, cheeks, nose, and chin, are recorded and written out in a CSV file. The function will create a new subdirectory under `output_dir` called `Color_Channel_Means/` where all CSV output files will be written.

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir` | `str` | A path string to the directory containing files to process. |
| `output_dir` | `str` | A path string to the directory where processed files will be output. |
| `color_space` | `int` or `str` | An integer specifier or string literal specifying the color space to operate in (One of COLOR_SPACE_RGB, COLOR_SPACE_HSV, COLOR_SPACE_GRAYSCALE, or "rgb", "hsv", "grayscale"). |
| `with_sub_dirs` | `bool` | A boolean flag indicating if the input directory contains sub-directories. |
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
# Define your input and output file paths
in_dir = "c:/my/path/to/files"
out_dir = "c:/my/path/to/output"

# Simplest call
pf.extract_face_color_means(in_dir, out_dir, pf.COLOR_SPACE_RGB)

# Nested subdirectories in input folder, HSV output
pf.extract_face_color_means(in_dir, out_dir, pf.COLOR_SPACE_HSV, With_sub_dirs=True)
```