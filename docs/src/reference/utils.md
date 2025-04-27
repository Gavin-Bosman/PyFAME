---
layout: doc
title: Utilities
prev: 
    text: 'Temporal Transforms'
    link: '/reference/temporal_transforms'
next: False
---
# Utilities Module Reference

## Retrieving Variable Names From Memory {#gen_utils}

The function `get_variable_name()` is frequently used within the package's internal logging calls. `get_variable_name()` takes in a variable of any type, and a namespace/scope (one of `locals()` or `globals()`), then returns the variables name in the given namespace as a string. 

```python
def get_variable_name(variable, namespace) -> str:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `variable` | `any` | A variable defined in the namespace provided. |
| `namespace` | `callable` | The namespace the variable occupies, one of `locals()` or `globals()`. |

## Compute The Angular Displacement Between Two Slopes

Utilized in many of the facial manipulations to aid in positional tracking, `calculate_rot_angle()` takes in two slopes, and returns the angular displacement in radians between the two slopes. The second slope parameter defaults to 0.0, which allows you to retreive the angular displacement from the x-axis. 

```python
def compute_rot_angle(slope1:float, slope2:float = 0.0) -> float:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `slope1` | `float` | The current slope value. |
| `slope2` | `float` | The previous slope value, or the x-axis if there is no previous slope. |

## Calculate The Intersection Point of Two Lines

Another utility function frequently used internally by the facial manipulation functions, in order to aid in positional tracking. `compute_line_intersection()` takes in two (x,y) pixel coordinates, and a line defined by an integer. If the path connecting the two points intersects with the provided line, the intersection point will be returned as a tuple containing the (x,y) pixel coordinates of the intersection. If no intersection point is found, `None` will be returned. The comparison line must be paralell to one of the axes; input parameter `vertical` is a boolean flag specifying the comparison case where the line is paralell to the y-axis, where it is paralell to the x-axis by default. 

```python
def compute_line_intersection(p1:tuple, p2:tuple, line:int, vertical:bool=False) -> tuple | None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `p1` | `tuple[int]` | The (x,y) integer coordinates of the first point. |
| `p2` | `tuple[int]` | The (x,y) integer coordinates of the second point. |
| `line` | `int` | An integer representing the equation of a line, that is paralell to one of the axes. |
| `vertical` | `bool` | A boolean flag indicating if the comparison line is a vertical line (paralell to the y-axis). |

### Error Handling

| Raises | Encountered Error |
| :----- | :---- |
| `ValueError` | Given unrecognized input parameter values. |
| `TypeError` | Given invalid input parameter typings. |

## Transcoding Input Video Files {#transcode}

PyFAME can operate over video files in the .mp4 and .mov formats, for any input video files encoded in another container, `transcode_video_to_mp4()` is your solution. This function leverages `cv2.VideoCapture()` to decode input video files, and `cv2.VideoWriter()` with fourcc encoding `*'mp4v'` to encode the videos to mp4. `transcode_video_to_mp4()` can transcode entire nested directories of videos in a single call, by providing the top-level directory path to `input_dir` and specifying `with_sub_dirs=True`.

```python
def transcode_video_to_mp4(input_dir:str, output_dir:str, with_sub_dirs:bool = False) -> None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir` | `str` | A path string to the directory containing files to process. |
| `output_dir` | `str` | A path string to the directory where processed files will be output. |
| `with_sub_dirs` | `bool` | A boolean flag indicating if the input directory contains nested subdirectories. |

### Error Handling

| Raises | Encountered Error |
| :----- | :---- |
| `ValueError` | Given unrecognized input parameter values. |
| `TypeError` | Given invalid input parameter typings. |
| `OSError` | Given invalid path-strings to the input or output directory. |

## Creating Custom Landmark Paths {#create_path}

Almost all of the facial manipulation functions take in an input paramater list of landmark paths. PyFAME contains many predefined landmark paths, but in the case a user may want to create their own custom path, they can use the `create_path()` function. This function takes in a list of integer landmark indicies (0-477) that align with MediaPipe's FaceMesh landmarks, and returns a list of tuples creating a closed path with the indicies provided. For example, if ['a', 'b', 'c'] was passed in, [('a', 'b'), ('b', 'c'), ('c', 'a')] would be returned. 

```Python
def create_path(landmark_set:list[int]) -> list[tuple]:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `landmark_set` | `list[int]` | A list of integer landmark id's that align with the MediaPipe FaceMesh landmarks. These landmark id's are integers in the range [0, 477]. | 