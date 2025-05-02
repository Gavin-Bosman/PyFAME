---
layout: doc
title: Analysis
prev: 
    text: 'Moviefy'
    link: '/reference/moviefy'
next: 
    text: 'Coloring'
    link: '/reference/coloring'
---
# Analysis Module Reference

## Optical Flow {#optic_flow}

`get_optical_flow` provides access to both sparse (Lucas-Kanadae) and dense (Farneback) optical flow. The type of optical flow operation performed is specified by input parameter `optical_flow_type`. For each file in `input_dir`, `get_optical_flow` applies the specified optical flow operation and outputs two files to `output_dir`. These two files are an image/video representation of the optical flow, and a CSV file containing the optical flow vector's magnitudes and direction. The input parameter `csv_sample_freq` determines how often the optical flow vectors should be sampled and recorded in the output CSV file. 

```Python
def get_optical_flow(
    input_dir:str, output_dir:str, optical_flow_type: int|str = SPARSE_OPTICAL_FLOW, 
    landmarks_to_track:list[int]|None = None, max_corners:int = 20, corner_quality_lvl:float = 0.3, min_corner_distance:int = 7, block_size:int = 5, win_size:tuple[int] = (15,15), 
    max_pyr_lvl:int = 2, pyr_scale:float = 0.5, max_iter:int = 10, lk_accuracy_thresh:float = 0.03, 
    poly_sigma:float = 1.2, point_color:tuple[int] = (255,255,255), point_radius:int = 5, 
    vector_color:tuple[int]|None = None, with_sub_dirs:bool = False, csv_sample_freq:int = 1000, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5
) -> None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir` | `str` | A path string to the directory containing files to process. |
| `output_dir` | `str` | A path string to the directory where processed files will be output. |
| `optical_flow_type` | `int` or `str` | An integer flag or string literal specifying the type of optical flow operation to perform (one of SPARSE_OPTICAL_FLOW/DENSE_OPTICAL_FLOW, or "sparse"/"dense"). |
| `landmarks_to_track` | `list[int]` or `None` | Specific landmark points of interest to pass to the Shi-Tomasi good corners algorithm. These integers must align with the mediapipe face landmarker landmarks, which range from 0-477. |
| `max_corners` | `int` | A configuration parameter specifying the maximum ammount of corners for the Shi-Tomasi corners algorithm to detect. |
| `corner_quality_lvl` | `float` | A float in the range [0,1] that determines the minimum quality of accepted corners found using the Shi-Tomasi corners algorithm. |
| `min_corner_distance` | `int` | The minimum euclidean distance between accepted corners found using the Shi-Tomasi corners algorithm. |
| `block_size` | `int` | The size of the search window used in the Shi-Tomasi corners algorithm (sparse optical flow), or the size of the pixel neighborhood used to find the polynomial expansion of each pixel (dense optical flow). `block_size` is commonly set equal to 5 or 7. |
| `win_size` | `tuple[int]` | The size of the search window (in pixels) used at each pyramid level with sparse optical flow. |
| `max_pyr_level` | `int` | The maximum number of pyramid levels used in sparse optical flow. As you increase this parameter larger motions can be detected but consequently computation time increases. |
| `pyr_scale` | `float` | A float in the range [0,1] representing the downscale of the image at each pyramid level in dense optical flow. For example, with a pyr_scale of 0.5, at each pyramid level the image will be half the size of the previous image. |
| `max_iter` | `int` | One of the termination criteria for both optical flow algorithms. Represents the maximum number of iterations over each frame the algorithm will make before terminating. |
| `lk_accuracy_thresh` | `float` | One of the termination criteria for sparse optical flow. A float in the range [0,1] representing the optimal termination accuracy for the algorithm. |
| `poly_sigma` | `float` | The standard deviation of the Gaussian distribution used in the polynomial expansion of each pixel for dense optical flow. Typically with `block_size` of 5 or 7, good values for `poly_sigma` are 1.2 and 1.5, respectively. |
| `point_color` | `tuple[int]` | A BGR color code that specifies the color of the points displayed in the sparse optical flow output image. |
| `point_radius` | `int` | The radius of the points displayed in the sparse optical flow output image. |
| `vector_color` | `tuple[int]` | A BGR color code that specifies the color of the vectors drawn in the sparse optical flow output image. |
| `with_sub_dirs`            | `bool` | A boolean flag indicating if the input directory contains sub-directories. |
| `csv_sample_freq` | `int` | The time duration (in msec) specifying the sampling period of the optical flow vectors magnitudes and directions. |
| `min_detection_confidence` | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `min_tracking_confidence`  | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |

### Error Handling
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

# Sparse optical flow, simplest call
pf.get_optical_flow(in_dir, out_dir, pf.SPARSE_OPTICAL_FLOW)

# Sparse optical flow, with input points
pf.get_optical_flow(in_dir, out_dir, pf.SPARSE_OPTICAL_FLOW, landmarks_to_track = [1,32,43,112,169])
# If you want to ONLY track the input points, set max_corners = len(landmark_list)

# Dense optical flow, simplest call
pf.get_optical_flow(in_dir, out_dir, pf.DENSE_OPTICAL_FLOW)

# Dense optical flow, get a more robust result with more iterations and greater search window
pf.get_optical_flow(in_dir, out_dir, pf.DENSE_OPTICAL_FLOW, max_iter=15, block_size=7, poly_sigma=1.5)

```

## Facial Color Means {#color_means}

`extract_face_color_means` will take each input image/video, and extract the global and local color channel values in the specified color space (determined by input parameter `color_space`). The color channel values of the full face, cheeks, nose, and chin, are recorded and written out in a CSV file. `extract_face_color_means` will create a new subdirectory under `output_dir` called `Color_Channel_Means/`.

```Python
def extract_face_color_means(
    input_dir:str, output_dir:str, color_space: int|str = COLOR_SPACE_RGB, 
    with_sub_dirs:bool = False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5
) -> None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir` | `str` | A path string to the directory containing files to process. |
| `output_dir` | `str` | A path string to the directory where processed files will be output. |
| `color_space` | `int` or `str` | An integer specifier or string literal specifying the color space to operate in (One of COLOR_SPACE_RGB, COLOR_SPACE_HSV, COLOR_SPACE_GRAYSCALE, or "rgb", "hsv", "grayscale"). |
| `with_sub_dirs` | `bool` | A boolean flag indicating if the input directory contains sub-directories. |
| `min_detection_confidence` | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `min_tracking_confidence`  | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |

### Error Handling
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