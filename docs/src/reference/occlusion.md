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
# Occlusion Module

## mask_face_region()

``` python
def mask_face_region(input_dir, output_dir, mask_type = FACE_OVAL_MASK)
```
Applies the specified `mask_type` to each input image or video file contained in `input_dir`.

The masked-out region of each image or frame is by default replaced with black (255,255,255), however other background colors can be specified by passing a BGR color code to input parameter `background_color`. This function creates a new output directory; outputting masked images and videos to {`output_dir`}/Masked.

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `str` | A path string to the directory containing files to process. |
| `output_dir`               | `str` | A path string to the directory where processed files will be output. |
| `mask_type`                | `int` | An integer flag specifying the type of masking operation being performed. The default value is `pyfame.FACE_OVAL_MASK`, for the complete list of options please see `pyfame.utils.display_face_mask_options()`. |
| `background_color`         | `tuple[int]` | A BGR color code specifying the output files background color. |
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
in_dir = "c:/my/path/to/input"
out_dir = "c:/my/path/to/output"

# Simplest call; which defaults to a face-oval mask with white background
pf.mask_face_region(in_dir, out_dir)

# Masking out the eyes, nose and mouth, with a black background
pf.mask_face_region(in_dir, out_dir, mask_type=pf.EYES_NOSE_MOUTH_MASK, background_color=(0,0,0))
```

## occlude_face_region()

``` python
occlude_face_region(input_dir, output_dir, landmarks_to_occlude = [BOTH_EYES_PATH])
```
For each input image or video contained in `input_dir`, the specified landmark paths are occluded. 

This function takes the landmark regions specified within `landmarks_to_occlude`, and occludes them with the specified `occlusion_fill`. `pyfame.OCCLUSION_FILL_BLACK` and `pyfame.OCCLUSION_FILL_MEAN` fill the exact shape of the provided landmark path with a solid color, while `pyfame.OCCLUSION_FILL_BAR` centers a horizontal rectangle over the provided landmark path. Occluded images and videos will be written to {`output_dir`}/Occluded.

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `str` | A path string to the directory containing files to process. |
| `output_dir`               | `str` | A path string to the directory where processed files will be output. |
| `landmarks_to_occlude` | `list[list[tuple]]` | One or more lists of facial landmark paths. These paths can be manually created using `pyfame.utils.utils.create_path()`, or you may use any of the library provided predefined landmark paths. |
| `occlusion_fill` | `int` | An integer flag indicating the occlusion method to be used. One of `OCCLUSION_FILL_BLACK`, `OCCLUSION_FILL_MEAN` or `OCCLUSION_FILL_BAR`. |
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
in_dir = "c:/my/path/to/input"
out_dir = "c:/my/path/to/output"

# Simplest call; which defaults to occluding the eyes with black
pf.occlude_face_region(in_dir, out_dir)

# Occluding the mouth with bar-style occlusion
pf.occlude_face_region(
    in_dir, out_dir, landmarks_to_occlude=[pf.MOUTH_PATH], 
    occlusion_fill=pf.OCCLUSION_FILL_BAR
)
```

## blur_face_region()
```python
blur_face_region(input_dir, output_dir, blur_method = "gaussian")
```
Applies a blur operation to each input image or video contained in `input_dir`.

`blur_face_region()` takes the provided `blur_method` (one of gaussian, average or median), and applies it to each image or video file contained in `input_dir`. The degree of blurring can be precisely controlled by manipulating the size of the blurring kernel, which can be specified by input parameter `k_size`. Blurred images and videos will be written out to {`output_dir`}/Blurred.

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `str` | A path string to the directory containing files to process. |
| `output_dir`               | `str` | A path string to the directory where processed files will be output. |
| `blur_method` | `str` or `int` | Either a string literal or an integer specifier of the blur method to be applied to the input files (one of "gaussian", "average" or "median"). |
| `k_size` | `int` | Specifies the size of the square kernel used in the blurring operations. |
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

```python
import pyfame as pf

# Define input paths
in_dir = "c:/my/path/to/input"
out_dir = "c:/my/path/to/output"

# Simplest call; defaults to guassian blur with a (15,15) kernel
pf.blur_face_region(in_dir, out_dir)

# Apply median-blur with (20,20) kernel
pf.blur_face_region(in_dir, out_dir, blur_method=pf.BLUR_METHOD_MEDIAN, k_size=20)
```

## apply_noise()

``` python
apply_noise(input_dir, output_dir, noise_method = "pixelate")
```
Applies the specified `noise_method` in the specified facial region to each input image or video contained in `input_dir`.

This function provides the ability to apply various types of image noise (pixelation, gaussian or salt and pepper) to dynamic facial regions using built-in facial region masks. `apply_noise` offers a high degree of noise customization, with parameters `noise_prob`, `mean`, `standard_dev` and `rand_seed` allowing for precise control over the manipulations and their reproducibility. Processed files will be written out to {`output_dir`}/Noise_Added.

### Parameters

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
in_dir = "c:/my/path/to/input"
out_dir = "c:/my/path/to/output"

# Simplest call; defaults to pixelation with pixel size of 32
pf.apply_noise(in_dir, out_dir)

# Applying salt and pepper noise, to just the facial skin
pf.apply_noise(
    in_dir, out_dir, noise_method = pf.NOISE_METHOD_SALT_AND_PEPPER,
    noise_prob=0.6, mask_type=pf.FACE_SKIN_MASK
)
```