import pytest
from pyfame.core.exceptions import *
from pyfame.core.temporal_transforms import shuffle_frame_order

def test_exception_handling():
    in_dir_valid = "tests\\data\\sample_video.mp4"
    in_dir_invalid = "tests\\data\\Videos\\Actor_01.mp4"
    out_dir_valid = "tests\\data\\outputs"
    out_dir_invalid = "tests\\images\\masked_videos"

    # exception testing input and output directories
    with pytest.raises(TypeError):
        shuffle_frame_order(input_dir=1, output_dir=out_dir_valid)
    with pytest.raises(OSError):
        shuffle_frame_order(input_dir=in_dir_invalid, output_dir=out_dir_valid)
    
    with pytest.raises(TypeError):
        shuffle_frame_order(input_dir=in_dir_valid, output_dir=1)
    with pytest.raises(OSError):
        shuffle_frame_order(input_dir=in_dir_valid, output_dir=out_dir_invalid)
    with pytest.raises(ValueError):
        shuffle_frame_order(input_dir=in_dir_valid, output_dir=in_dir_valid)

    # exception testing shuffle_method parameter
    with pytest.raises(TypeError):
        shuffle_frame_order(input_dir=in_dir_valid, output_dir=out_dir_valid, shuffle_method="test")
    with pytest.raises(ValueError):
        shuffle_frame_order(input_dir=in_dir_valid, output_dir=out_dir_valid, shuffle_method=100)
    
    # exception testing rand_seed parameter
    with pytest.raises(TypeError):
        shuffle_frame_order(input_dir=in_dir_valid, output_dir=out_dir_valid, rand_seed=2.5)
    
    # exception testing block_duration parameter
    with pytest.raises(TypeError):
        shuffle_frame_order(input_dir=in_dir_valid, output_dir=out_dir_valid, block_duration="test")
    with pytest.raises(ValueError):
        shuffle_frame_order(input_dir=in_dir_valid, output_dir=out_dir_valid, block_duration=30)

    # exception testing block_order parameter
    with pytest.raises(TypeError):
        shuffle_frame_order(input_dir=in_dir_valid, output_dir=out_dir_valid, block_order=["A", "b"])
    with pytest.raises(ValueError):
        shuffle_frame_order(input_dir=in_dir_valid, output_dir=out_dir_valid, block_order=({"a":1}, 2))
    with pytest.raises(ValueError):
        shuffle_frame_order(input_dir=in_dir_valid, output_dir=out_dir_valid, block_order=([1,2,3], 2.5))
    
    # exception testing drop_last_block parameter
    with pytest.raises(TypeError):
        shuffle_frame_order(input_dir=in_dir_valid, output_dir=out_dir_valid, drop_last_block="test")
    
    # exception testing with_sub_dirs parameter
    with pytest.raises(TypeError):
        shuffle_frame_order(input_dir=in_dir_valid, output_dir=out_dir_valid, with_sub_dirs="test")
    
    # testing the passing of unrecognized file extenstions
    with pytest.raises(UnrecognizedExtensionError):
        shuffle_frame_order(input_dir="tests\\data\\invalid_ext.webp", output_dir=out_dir_valid)