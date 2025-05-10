---
layout: doc
title: Temporal Transforms
prev: 
    text: 'Scrambling'
    link: '/reference/scrambling'
next: 
    text: 'Utilities'
    link: '/reference/utils'
---
# Temporal Transforms Module

## shuffle_frame_order()

```python
shuffle_frame_order(input_dir, output_dir, shuffle_method = FRAME_SHUFFLE_RANDOM)
```
Breaks up the input video(s) frames into 'blocks' of equal size, then positionally shuffles them according to the specified `shuffle_method`.

`shuffle_frame_order()` provides users the ability to temporally segment groups of frames in a video (in 'blocks'), and shuffle their ordering. The function can perform 6+ frame shuffling operations, ranging from random reassignment to cyclic shuffling to frame-interleaving. The `block_order` input parameter expects a tuple containing a zero-indexed list specifying the block ordering, and an integer specifying the number of frames per block. Both of these values can be provided manually, but the parameter is set up in this way to accept the output of helper function `generate_shuffled_block_array()`. The total number of frames in the input video often will not evenly divide into a set number of blocks, thus the `drop_last_block` parameter can be passed to drop the last uneven block from the output video.

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir` | `str` | A path string to the directory containing files to process. |
| `output_dir` | `str` | A path string to the directory where processed files will be output. |
| `shuffle_method` | `int` | An integer flag specifying the frame shuffling method to be used. The default is `pyfame.FRAME_SHUFFLE_RANDOM`, for a full list of shuffling options see `pyfame.utils.display_shuffle_method_options()`. |
| `rand_seed` | `int` or `None` | An integer seed to the random number generator used in the frame shuffling operations. |
| `block_order` | `tuple` or `None` | A tuple containing a zero-indexed list of integers specifying the block ordering, and an integer specifying the number of frames per block. These can be manually provided or generated and directly passed from `generate_shuffled_block_array()`. |
| `block_duration` | `int` | The time duration in milliseconds of each temporal block. The exact frame count of each block depends on the input video's fps, but an example duration of 500ms with a 30fps video would result in ~15 frames per temporal block. |
| `drop_last_block` | `bool` | A boolean flag specifying if the last (uneven sized) block should be dropped from the output video. |
| `with_sub_dirs` | `bool` | A boolean flag specifying if the input directory contains nested subdirectories. |

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

### Quick Example

```Python
import pyfame as pf

# Define input paths
in_dir = "c:/my/path/to/input/"
out_dir = "c:/my/path/to/output/"

# Simplest call; defaults to fully random frame shuffling 
# with 1 second temporal blocks
pf.shuffle_frame_order(in_dir, out_dir, rand_seed = 1234)

# Left cyclic shift, with temporal blocks of 500 msec,
# dropping the last uneven block
pf.shuffle_frame_order(
    in_dir, out_dir, shuffle_method = pf.FRAME_SHUFFLE_LEFT_CYCLIC_SHIFT, 
    rand_seed = 1234, block_duration = 500, drop_last_block = True
)
```

## generate_shuffled_block_array()

```python
generate_shuffled_block_array(file_path, shuffle_method = FRAME_SHUFFLE_RANDOM)
```
Precomputes the `block_size` and block-ordering array for the specified `shuffle_method`, then returns them as a tuple.

`generate_shuffled_block_array()` is a helper function to `shuffle_frame_order()` that allows users to provide an input video, a shuffling method, and a temporal block duration, which returns a tuple containing (block order array, num frames per block). This function was created in order to abstract the `block_size` (num frames per block) from the user, as determining it's value was not very intuitive. Instead, using the `block_duration` and `shuffle_method`, `generate_shuffled_block_array()` automatically computes the optimal `block_size`, then uses it to generate a block order array, shuffled using the method of choice. The output block order array is a zero-indexed array, that determines the order that the frame-blocks will be written out in `shuffle_frame_order()`. Additionally, in order to ease the processing pipeline, the output of `generate_shuffled_block_array()` can be directly passed as an input parameter `block_order` to the `shuffle_Frame_order()` function. 

### Parameters

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `file_path` | `str` | The path string to the input video file. |
| `shuffle_method` | `int` | An integer flag specifying the frame shuffling method to be used. The default is `pyfame.FRAME_SHUFFLE_RANDOM`, for a full list of shuffling options see `pyfame.utils.display_shuffle_method_options()`. |
| `rand_seed` | `int` or `None` | An integer seed to the random number generator used in the frame shuffling operations. |
| `block_duration` | `int` | The time duration in milliseconds of each temporal block. The exact frame count of each block depends on the input video's fps, but an example duration of 500ms with a 30fps video would result in ~15 frames per temporal block. |

### Returns

| Type | Description |
| :--- | :---------- |
| `tuple[list, int]` | A tuple containing the block-ordering list, and a block_size integer. |

### Exceptions Raised

| Raises | Encountered Error |
| :----- | :---- |
| `ValueError` | Given unrecognized input parameter values. |
| `TypeError` | Given invalid input parameter typings. |
| `OSError` | Given an invalid path-string for either `input_dir` or `output_dir`. |
| `FileReadError` | If an error is encountered instantiating `cv2.VideoCapture()` or calling `cv2.imRead()`. |

### Quick Example

```Python
import pyfame as pf

# Define input paths
in_dir = "c:/my/path/to/input/"
out_dir = "c:/my/path/to/output/"

# Simplest call; defaults to fully random frame shuffling 
# with 1 second temporal blocks
pf.generate_shuffled_block_array(in_dir, rand_seed = 1234)

# This function can also be used as a preprocessing step that directly feeds
# it's output into shuffle_frame_order()
blocks = pf.generate_shuffled_block_array(
    in_dir, shuffle_method = pf.FRAME_SHUFFLE_PALINDROME, rand_seed = 1234
)

# We can then feed block_arr directly as input to shuffle_frame_order()
pf.shuffle_frame_order(in_dir, out_dir, block_order = blocks, drop_last_block = True)
```