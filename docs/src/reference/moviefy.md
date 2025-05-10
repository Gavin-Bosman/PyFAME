---
layout: doc
title: Moviefy
prev: 
    text: 'Overview'
    link: '/reference/overview'
next: 
    text: 'Analysis'
    link: '/reference/analysis'
---
# Moviefy Module

## equate_image_sizes()

```Python
equate_image_sizes(input_dir, method = EQUATE_IMAGES_CROP)
```
Equalizes the image dimensions of all the images contained in the input directory provided.

The function leverages two equalization methods; `pyfame.EQUATE_IMAGES_CROP` and `pyfame.EQUATE_IMAGES_PAD`. Prior to equalization, the function scans through the content of the input directory to find the maximal and minimal image dimensions contained within it. Then, depending on the `method` specified, each image will be either cropped down to, or padded up to, equivalent dimensions. When equalizing image sizes with padding, the input parameter `pad_color` specifies what color to fill the padded area with. 

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir` | `str` | The path string to the directory containing all images to be normalized. |
| `method` | `int` | An integer flag specifying the normalization method to use; one of `pyfame.EQUATE_IMAGES_CROP` or `pyfame.EQUATE_IMAGES_PAD`. |
| `pad_color` | `tuple[int]` | A tuple containing an integer BGR color code specifying the color of the padded border. The default value is white or (255,255,255). |
| `with_sub_dirs` |  `bool` | A boolean flag indicating if the input directory contains nested subdirectories. |

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

### Quick Example

```Python
import pyfame as pf

# Define input paths
in_dir = "c:/my/path/to/input/"

# Performing img size normalization with cropping
pf.equate_image_sizes(in_dir, method=pf.EQUATE_IMAGES_CROP)

# OR

# Performing img size normalization with padding
pf.equate_image_sizes(in_dir, method=pf.EQUATE_IMAGES_PAD, pad_color=(0,0,0))
```

## moviefy_images()

```Python
moviefy_images(input_dir, output_dir, output_filename, equalization_method = EQUATE_IMAGES_CROP)
```
Converts a series of static images into a 'movie' by repeating and interpolating frames.

`moviefy_images()` repeats and interpolates the input images as video frames, and writes them out to an mp4 video. The time duration that each image will repeat for is determined by input parameter `repeat_duration`. Furthermore, if the user wants successive images to have an interpolated (blended) transition, a proportion of the repeat duration will be used to transition into the next image. This proportion is determined by input parameter `blended_frames_prop`. `moviefy_images()` requires all input images to be of a standard size, thus `equate_image_sizes()` can be used either as a preprocessing step, or normalization can be performed internally by setting input parameter `equate_sizes = True`.

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir` | `str` | A path string to the directory containing images to moviefy. |
| `output_dir` | `str` | A path string to the directory where the output video will be written to. |
| `output_filename` | `str` | The file name for the output video. |
| `fps` | `int` | The frame rate of the output video (in frames/second). | 
| `repeat_duration` | `int` | A time duration in msec, specifying how long each image will be repeated for in the output video. |
| `blend_images` | `bool` | A boolean flag specifying if successive images should have an interpolated (blended) transition. |
| `blended_frames_prop` | `float` | A float value in the range [0,1] specifying the proportion of each images repeat duration dedicated to interpolating successive images. |
| `equate_sizes` | `bool` | A boolean flag indicating if the input images need to have their dimensions equalized. |
| `equalization_method` | `int` | An integer flag specifying the equalization method to use; one of `pyfame.EQUATE_IMAGES_CROP` or `pyfame.EQUATE_IMAGES_PAD`. |
| `pad_color` | `tuple[int]` | A tuple containing an integer BGR color code, specifying the color of the padded border added to the images. The default value is white or (255,255,255). |
| `with_sub_dirs` |  `bool` | A boolean flag indicating if the input directory contains nested subdirectories. |

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
| `ImageShapeError` | If any of the input images have mismatching dimensions. |

### Quick Example

```Python
import pyfame as pf

# Define input paths
in_dir = "c:/my/path/to/input/"
out_dir = "c:/my/path/to/output/"

# Simplest Call
pf.moviefy_images(in_dir, out_dir, output_filename="movie_1")

# Moviefy with image dimension normalization
pf.moviefy_images(
    in_dir, out_dir, output_filename="movie_2", 
    equate_sizes=True, equalization_method=pf.EQUATE_IMAGES_CROP
)

# Moviefy with frame interpolation
pf.moviefy_images(
    in_dir, out_dir, output_filename="movie_3",
    repeat_duration=800, blend_images=True, blended_frames_prop=0.2
)
```
