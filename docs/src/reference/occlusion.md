---
layout: doc
title: Occlusion
prev: 
    text: 'Coloring'
    link: '/reference/coloring'
next: 
    text: 'Point-Light Display'
    link: '/reference/pld'
---
# Occlusion Module Reference

## Facial Masking {#facial_masking}

`mask_face_region` will apply the specified `mask_type` to all files contained in `input_dir`, outputting masked images and videos to {`output_dir`}/Masked.

``` python
def mask_face_region(
    input_dir:str, output_dir:str, mask_type:int = FACE_OVAL_MASK, 
    with_sub_dirs:bool = False, background_color: tuple[int] = (255,255,255),
    min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5
) -> None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `str` | A path string to the directory containing files to process. |
| `output_dir`               | `str` | A path string to the directory where processed files will be output. |
| `mask_type`                | `int` | An integer flag specifying the type of masking operation being performed. One of `FACE_OVAL`, `FACE_OVAL_TIGHT` OR `FACE_SKIN_ISOLATION`. |
| `background_color`         | `tuple[int]` | A BGR color code specifying the output files background color. |
| `with_sub_dirs`            | `bool` | A boolean flag indicating if the input directory contains sub-directories. |
| `min_detection_confidence` | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `min_tracking_confidence`  | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |

### Error Handling {#facial_masking_error}

| Raises | Encountered Error |
| :----- | :---- |
| `ValueError` | Given unrecognized input parameter values. |
| `TypeError` | Given invalid input parameter typings. |
| `OSError` | Given an invalid path-string for either `input_dir` or `output_dir`. |
| `FileReadError` | If an error is encountered instantiating `cv2.VideoCapture()` or calling `cv2.imRead()`. |
| `FileWriteError` | If an error is encountered instantiating `cv2.VideoWriter()` or calling `cv2.imWrite()`. |
| `UnrecognizedExtensionError` | If the function encounters an unrecognized image or video file extension. |
| `FaceNotFoundError` | If the mediapipe FaceMesh model cannot identify a face in the input image or video. |

## Facial Occlusion {#facial_occlusion}

`occlude_face_region` takes the landmark regions specified within `landmarks_to_occlude`, and occludes them with the specified method for each image or video file present within the input directory provided in `input_dir`. Processed videos will be written to {`output_dir`}/Occluded.

``` python
def occlude_face_region(
    input_dir:str, output_dir:str, landmarks_to_occlude:list[list[tuple]] = [BOTH_EYES_PATH],
    occlusion_fill:int = OCCLUSION_FILL_BLACK, with_sub_dirs:bool =  False, 
    min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5
) -> None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `str` | A path string to the directory containing files to process. |
| `output_dir`               | `str` | A path string to the directory where processed files will be output. |
| `landmarks_to_occlude` | `list[list[tuple]]` | One or more lists of facial landmark paths. These paths can be manually created using `pyfame.utils.utils.create_path()`, or you may use any of the library provided predefined landmark paths. |
| `occlusion_fill` | `int` | An integer flag indicating the occlusion method to be used. One of `OCCLUSION_FILL_BLACK`, `OCCLUSION_FILL_MEAN` or `OCCLUSION_FILL_BAR`. |
| `with_sub_dirs`            | `bool` | A boolean flag indicating if the input directory contains sub-directories. |
| `min_detection_confidence` | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `min_tracking_confidence`  | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |

### Error Handling {#facial_occlusion_error}

| Raises | Encountered Error |
| :----- | :---- |
| `ValueError` | Given unrecognized input parameter values. |
| `TypeError` | Given invalid input parameter typings. |
| `OSError` | Given an invalid path-string for either `input_dir` or `output_dir`. |
| `FileReadError` | If an error is encountered instantiating `cv2.VideoCapture()` or calling `cv2.imRead()`. |
| `FileWriteError` | If an error is encountered instantiating `cv2.VideoWriter()` or calling `cv2.imWrite()`. |
| `UnrecognizedExtensionError` | If the function encounters an unrecognized image or video file extension. |
| `FaceNotFoundError` | If the mediapipe FaceMesh model cannot identify a face in the input image or video. |

## Facial Blurring {#facial_blurring}

`blur_face_region` takes the provided `blur_method` (one of gaussian, average or median), and applies it to each image or video file contained in `input_dir`. The processed files are written out to {`output_dir`}/Blurred.

```python
def blur_face_region(
    input_dir:str, output_dir:str, blur_method:str | int = "gaussian", k_size:int = 15, 
    with_sub_dirs:bool = False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5
) -> None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `str` | A path string to the directory containing files to process. |
| `output_dir`               | `str` | A path string to the directory where processed files will be output. |
| `blur_method` | `str` or `int` | Either a string literal or an integer specifier of the blur method to be applied to the input files (one of "gaussian", "average" or "median"). |
| `k_size` | `int` | Specifies the size of the square kernel used in the blurring operations. |
| `with_sub_dirs`            | `bool` | A boolean flag indicating if the input directory contains sub-directories. |
| `min_detection_confidence` | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `min_tracking_confidence`  | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |

### Error Handling {#facial_blurring_error}

| Raises | Encountered Error |
| :----- | :---- |
| `ValueError` | Given unrecognized input parameter values. |
| `TypeError` | Given invalid input parameter typings. |
| `OSError` | Given an invalid path-string for either `input_dir` or `output_dir`. |
| `FileReadError` | If an error is encountered instantiating `cv2.VideoCapture()` or calling `cv2.imRead()`. |
| `FileWriteError` | If an error is encountered instantiating `cv2.VideoWriter()` or calling `cv2.imWrite()`. |
| `UnrecognizedExtensionError` | If the function encounters an unrecognized image or video file extension. |
| `FaceNotFoundError` | If the mediapipe FaceMesh model cannot identify a face in the input image or video. |

## Dynamic Noise Application {#facial_noise}

`apply_noise` provides the ability to apply various types of image noise (pixelation, gaussian or salt and pepper) to dynamic facial regions using built-in facial region masks. `apply_noise` offers a high degree of noise customization, with parameters `noise_prob`, `mean`, `standard_dev` and `rand_seed` allowing for precise control over the manipulations and their reproducibility. Processed files are written out to {`output_dir`}/Noise_Added.

``` python
def apply_noise(
    input_dir:str, output_dir:str, noise_method:str|int = "pixelate", pixel_size:int = 32, 
    noise_prob:float = 0.5, rand_seed:int | None = None, mean:float = 0.0, standard_dev:float = 0.5, 
    mask_type:int = FACE_OVAL_MASK, with_sub_dirs:bool = False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5
) -> None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `str` | A path string to the directory containing files to process. |
| `output_dir`               | `str` | A path string to the directory where processed files will be output. |
| `noise_method` | `int` or `str` | Either a string literal or an integer specifier specifying the noise method to be applied (one of "pixelate", "salt and pepper" or "gaussian"). |
| `pixel_size` | `int` | Specifies the pixel size to be used with noise_method "pixelate". |
| `noise_prob` | `float` | A float in the range [0,1] that specifies the probability of noise being applied to a random pixel in the frame. |
| `rand_seed` | `int` | An integer to seed the random number generator. |
| `mean` | `float` | The mean of the gaussian distribution used with noise_method "gaussian". |
| `standard_dev` | `float` | The standard deviation of the gaussian distribution used with noise_method "gaussian". |
| `mask_type`                | `int` | An integer flag specifying the type of masking operation being performed. |
| `with_sub_dirs`            | `bool` | A boolean flag indicating if the input directory contains sub-directories. |
| `min_detection_confidence` | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `min_tracking_confidence`  | `float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |

### Error Handling {#facial_noise_error}

| Raises | Encountered Error |
| :----- | :---- |
| `ValueError` | Given unrecognized input parameter values. |
| `TypeError` | Given invalid input parameter typings. |
| `OSError` | Given an invalid path-string for either `input_dir` or `output_dir`. |
| `FileReadError` | If an error is encountered instantiating `cv2.VideoCapture()` or calling `cv2.imRead()`. |
| `FileWriteError` | If an error is encountered instantiating `cv2.VideoWriter()` or calling `cv2.imWrite()`. |
| `UnrecognizedExtensionError` | If the function encounters an unrecognized image or video file extension. |
| `FaceNotFoundError` | If the mediapipe FaceMesh model cannot identify a face in the input image or video. |