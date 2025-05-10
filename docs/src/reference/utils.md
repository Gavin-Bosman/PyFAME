---
layout: doc
title: Utilities
prev: 
    text: 'Temporal Transforms'
    link: '/reference/temporal_transforms'
next: 
    text: 'Codebook'
    link: '/reference/codebook'
---
# Utilities Module

## get_variable_name()

```python
get_variable_name(variable, namespace)
```
Retrieve a variables name in the specified namespace.

The function `get_variable_name()` is frequently used within the package's internal logging calls. `get_variable_name()` takes in a variable of any type, and a namespace/scope (one of `locals()` or `globals()`), then returns the variables name in the given namespace as a string. 

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `variable` | `any` | A variable defined in the namespace provided. |
| `namespace` | `callable` | The namespace the variable occupies, one of `locals()` or `globals()`. |

### Returns

| Type | Description |
| :--- | :---------- |
| `str` | A string literal containing the variables name in the specified namespace. |

### Quick Example

```Python
import pyfame as pf
import pyfame.utils as pfu

my_variable = "text"

# Get_variable_name() is used to return the literal variable name in 
# the provided scope as a string.
pfu.get_variable_name(my_variable, locals())    #--> "my_variable"

pfu.get_variable_name(pf.FACE_OVAL_MASK, globals()) #--> "FACE_OVAL_MASK"
```

## compute_rot_angle()

```python
compute_rot_angle(slope1:float, slope2:float = 0.0)
```
Calculates and returns the angular difference (in radians) between two slopes.

Utilized in many of the facial manipulations to aid in positional tracking, `calculate_rot_angle()` takes in two slopes, and returns the angular displacement in radians between the two slopes. The second slope parameter defaults to 0.0, which allows you to retreive the angular displacement from the x-axis. 

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `slope1` | `float` | The current slope value. |
| `slope2` | `float` | The previous slope value, or the x-axis if there is no previous slope. |

### Returns

| Type | Description |
| :--- | :---------- |
| `float` | The angular difference (in radians) between slope2 and slope1. |

### Quick Example

```Python
import pyfame.utils as pfu

slope_1 = 2/3
slope_2 = 1/3

# Get the angular displacement from the x-axis
pfu.compute_rot_angle(slope1=slope_1)

# Get the angular displacement from a previous slope
pfu.compute_rot_angle(slope1=slope_1, slope2=slope_2)
```

## compute_line_intersection()

```python
compute_line_intersection(p1, p2, line)
```
Compute and return the intersection point between the line formed by connecting `p1` and `p2`, and the line defined by `line`.

Another utility function frequently used internally by the facial manipulation functions, in order to aid in positional tracking. `compute_line_intersection()` takes in two (x,y) pixel coordinates, and a line defined by an integer. If the path connecting the two points intersects with the provided line, the intersection point will be returned as a tuple containing the (x,y) pixel coordinates of the intersection. If no intersection point is found, `None` will be returned. The comparison line must be paralell to one of the axes; input parameter `vertical` is a boolean flag specifying the comparison case where the line is paralell to the y-axis, where it is paralell to the x-axis by default. 

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `p1` | `tuple[int]` | The (x,y) integer coordinates of the first point. |
| `p2` | `tuple[int]` | The (x,y) integer coordinates of the second point. |
| `line` | `int` | An integer representing the equation of a line, that is paralell to one of the axes. |
| `vertical` | `bool` | A boolean flag indicating if the comparison line is a vertical line (paralell to the y-axis). |

### Returns

| Type | Description |
| :--- | :---------- |
| `tuple` or `None` | If an intersection point is found, its (x,y) pixel coordinates are returned as a tuple of integers, otherwise `None` is returned. |

### Exceptions Raised

| Raises | Encountered Error |
| :----- | :---- |
| `ValueError` | Given unrecognized input parameter values. |
| `TypeError` | Given invalid input parameter typings. |

### Quick Example

```Python
import pyfame.utils as pfu

p1 = (12, 127)
p2 = (72, 85)
line = 50

# Retrieve (x,y) intersection point with x=50
pfu.compute_line_intersection(p1, p2, line, vertical=True)  #--> (50,100)
```

## transcode_video_to_mp4()

```python
transcode_video_to_mp4(input_dir, output_dir)
```
Transcode some alternative video container (.avi, .mov, etc.) to an mp4 container.

PyFAME can operate over video files in the .mp4 and .mov formats, for any input video files encoded in another container, `transcode_video_to_mp4()` is your solution. This function leverages `cv2.VideoCapture()` to decode input video files, and `cv2.VideoWriter()` with fourcc encoding `*'mp4v'` to encode the videos to mp4. `transcode_video_to_mp4()` can transcode entire nested directories of videos in a single call, by providing the top-level directory path to `input_dir` and specifying `with_sub_dirs=True`.

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir` | `str` | A path string to the directory containing files to process. |
| `output_dir` | `str` | A path string to the directory where processed files will be output. |
| `with_sub_dirs` | `bool` | A boolean flag indicating if the input directory contains nested subdirectories. |

### Returns

`None`

### Exceptions Raised

| Raises | Encountered Error |
| :----- | :---- |
| `ValueError` | Given unrecognized input parameter values. |
| `TypeError` | Given invalid input parameter typings. |
| `OSError` | Given invalid path-strings to the input or output directory. |

### Quick Example

```Python
import pyfame.utils as pfu

# Define input paths
in_dir = "c:/my/path/to/video/file.avi"
out_dir = "c:/my/path/to/output"

# Transcode .avi to .mp4
pfu.transcode_video_to_mp4(in_dir, out_dir)
```

## create_path()

```Python
create_path(landmark_set)
```
Given a list of integer landmark indicies, return a closed path in the form [(a, b), (b, c), ..., (y, z), (z, a)].

Almost all of the facial manipulation functions take in an input paramater list of landmark paths. PyFAME contains many predefined landmark paths, but in the case a user may want to create their own custom path, they can use the `create_path()` function. This function takes in a list of integer landmark indicies (0-477) that align with MediaPipe's FaceMesh landmarks, and returns a list of tuples creating a closed path with the indicies provided. For example, if ['a', 'b', 'c'] was passed in,  would be returned. 

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `landmark_set` | `list[int]` | A list of integer landmark id's that align with the MediaPipe FaceMesh landmarks. These landmark id's are integers in the range [0, 477]. | 

### Returns

| Type | Description |
| :--- | :---------- |
| `list[tuple[int]]` | A list of tuples containing integers that form a closed path. |

### Quick Example

```Python
import pyfame.utils as pfu

# Define a landmark set
lms = [1, 12, 20, 16]

# Create a path variable
my_path = pfu.create_path(lms)  #--> [(1,12), (12,20), ..., (16,1)]
```