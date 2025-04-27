---
layout: doc
title: Coloring
prev:
    text: 'Analysis'
    link: '/reference/analysis'
next: 
    text: 'Occlusion'
    link: '/reference/occlusion'
---
# Coloring Module Reference

## Facial Color Manipulation {#face_color_shift}

`face_color_shift` performs a weighted color shift on the color channel specified in `shift_color`, weighted by the outputs of the provided `timing_func`. Parameters `onset_t` and `offset_t` can be used to specify when the color shifting fades in and fades out (this only applies to video files). `face_color_shift` makes use of the CIELAB color space to perform color shifting, due to it being far more perceptually uniform than the standard RGB or BGR color spaces.

::: warning
`face_color_shift` requires the outputs of the provided `timing_func` to be normalised; that is, in the range [0,1]. Predefined normalised functions such as `sigmoid`, `linear`, `gaussian` and `constant` are available for use in `pyfame.utils`. Extra parameters for these functions can be passed to `face_color_shift` as keyword arguments. Users may also define their own timing functions, but it is up to the user to ensure their functions take at least one input float parameter, and that the return value is within the normal range.
:::
    
```python
def face_color_shift(
    input_dir:str, output_dir:str, onset_t:float = 0.0, offset_t:float = 0.0, 
    shift_magnitude: float = 8.0, timing_func:Callable[...,float] = linear, 
    shift_color:str|int = COLOR_RED, landmark_regions:list[list[tuple]] = [FACE_SKIN_PATH], 
    with_sub_dirs:bool = False, min_detection_confidence:float = 0.5, 
    min_tracking_confidence:float = 0.5, **kwargs
) -> None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `str` | A path string to the directory containing files to process. |
| `output_dir`               | `str` | A path string to the directory where processed files will be output. |
| `onset_t`                  | `float` | The onset time when colour shifting will begin. |
| `offset_t`                 | `float` | The offset time when colour shifting will begin to fade out. |
| `shift_magnitude`          | `float` | The maximum units to shift the specified colour channel by, during peak onset. |
| `timing_func`              | `Callable[..., float]` | Any function that takes at least one float, and returns a normalised float value. |
| `landmark_regions`         | `list[list[tuple]]` | A list of one or more landmark paths, specifying the region in which the colouring will take place. |
| `shift_colour`             | `str or int` | Either a string literal (i.e. "red"), or a predefined integer constant; one of `COLOR_RED`, `COLOR_BLUE`, `COLOR_GREEN` or `COLOR_YELLOW`. |
| `with_sub_dirs`            | `bool` | A boolean flag indicating if the input directory contains sub-directories. |
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

## Facial Saturation Manipulation {#face_sat_shift}

`face_saturation_shift` performs a weighted saturation shift in a near identical manner to `face_color_shift`. This function makes use of the HSV color space to manipulate saturation. 

```python
def face_saturation_shift(
    input_dir:str, output_dir:str, onset_t:float = 0.0, offset_t:float = 0.0, 
    shift_magnitude:float = -8.0, timing_func:Callable[..., float] = linear, 
    landmark_regions:list[list[tuple]] = [FACE_SKIN_PATH], with_sub_dirs:bool = False, 
    min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5, **kwargs
) -> None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `str` | A path string to the directory containing files to process. |
| `output_dir`               | `str` | A path string to the directory where processed files will be output. |
| `onset_t`                  | `float` | The onset time when colour shifting will begin. |
| `offset_t`                 | `float` | The offset time when colour shifting will begin to fade out. |
| `shift_magnitude`          | `float` | The maximum units to shift the specified colour channel by, during peak onset. |
| `timing_func`              | `Callable[..., float]` | Any function that takes at least one float, and returns a normalised float value. |
| `landmark_regions`         | `list[list[tuple]]` | A list of one or more landmark paths, specifying the region in which the saturation shift will take place. |
| `with_sub_dirs`            | `bool` | A boolean flag indicating if the input directory contains sub-directories. |
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

## Facial Brightness Manipulation {#face_bright_shift}

`face_brightness_shift` performs a weighted brightness shift in a near identical manner to `face_color_shift`. This function leverages native OpenCV operations like `cv2.convertScaleAbs` to manipulate image brightness.

```python
def face_brightness_shift(
    input_dir:str, output_dir:str, onset_t:float = 0.0, offset_t:float = 0.0, 
    shift_magnitude:float = 20.0, timing_func:Callable[..., float] = linear, 
    landmark_regions:list[list[tuple]] = [FACE_SKIN_PATH], with_sub_dirs:bool = False, 
    min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5, **kwargs
) -> None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `str` | A path string to the directory containing files to process. |
| `output_dir`               | `str` | A path string to the directory where processed files will be output. |
| `onset_t`                  | `float` | The onset time when colour shifting will begin. |
| `offset_t`                 | `float` | The offset time when colour shifting will begin to fade out. |
| `shift_magnitude`          | `float` | The maximum units to shift the specified colour channel by, during peak onset. |
| `timing_func`              | `Callable[..., float]` | Any function that takes at least one float, and returns a normalised float value. |
| `landmark_regions`         | `list[list[tuple]]` | A list of one or more landmark paths, specifying the region in which the brightness shift will take place. |
| `with_sub_dirs`            | `bool` | A boolean flag indicating if the input directory contains sub-directories. |
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