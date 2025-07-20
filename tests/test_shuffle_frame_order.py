import pytest
from pyfame.utilities.exceptions import *
from pyfame.manipulation.temporal.apply_temporal_frame_shuffle import apply_frame_shuffle

def test_exception_handling():
    in_dir_valid = "tests\\data\\sample_video.mp4"
    in_dir_invalid = "tests\\data\\Videos\\Actor_01.mp4"
    out_dir_valid = "tests\\data\\outputs"
    out_dir_invalid = "tests\\images\\masked_videos"

    # exception testing input and output directories
    with pytest.raises(TypeError):
        apply_frame_shuffle(input_directory=1, output_directory=out_dir_valid)
    with pytest.raises(OSError):
        apply_frame_shuffle(input_directory=in_dir_invalid, output_directory=out_dir_valid)
    
    with pytest.raises(TypeError):
        apply_frame_shuffle(input_directory=in_dir_valid, output_directory=1)
    with pytest.raises(OSError):
        apply_frame_shuffle(input_directory=in_dir_valid, output_directory=out_dir_invalid)
    with pytest.raises(ValueError):
        apply_frame_shuffle(input_directory=in_dir_valid, output_directory=in_dir_valid)

    # exception testing shuffle_method parameter
    with pytest.raises(TypeError):
        apply_frame_shuffle(input_directory=in_dir_valid, output_directory=out_dir_valid, shuffle_method="test")
    with pytest.raises(ValueError):
        apply_frame_shuffle(input_directory=in_dir_valid, output_directory=out_dir_valid, shuffle_method=100)
    
    # exception testing rand_seed parameter
    with pytest.raises(TypeError):
        apply_frame_shuffle(input_directory=in_dir_valid, output_directory=out_dir_valid, rand_seed=2.5)
    
    # exception testing block_duration parameter
    with pytest.raises(TypeError):
        apply_frame_shuffle(input_directory=in_dir_valid, output_directory=out_dir_valid, block_duration="test")
    with pytest.raises(ValueError):
        apply_frame_shuffle(input_directory=in_dir_valid, output_directory=out_dir_valid, block_duration=30)

    # exception testing block_order parameter
    with pytest.raises(TypeError):
        apply_frame_shuffle(input_directory=in_dir_valid, output_directory=out_dir_valid, block_order=["A", "b"])
    with pytest.raises(ValueError):
        apply_frame_shuffle(input_directory=in_dir_valid, output_directory=out_dir_valid, block_order=({"a":1}, 2))
    with pytest.raises(ValueError):
        apply_frame_shuffle(input_directory=in_dir_valid, output_directory=out_dir_valid, block_order=([1,2,3], 2.5))
    
    # exception testing drop_last_block parameter
    with pytest.raises(TypeError):
        apply_frame_shuffle(input_directory=in_dir_valid, output_directory=out_dir_valid, drop_last_block="test")
    
    # exception testing with_sub_dirs parameter
    with pytest.raises(TypeError):
        apply_frame_shuffle(input_directory=in_dir_valid, output_directory=out_dir_valid, with_sub_dirs="test")
    
    # testing the passing of unrecognized file extenstions
    with pytest.raises(UnrecognizedExtensionError):
        apply_frame_shuffle(input_directory="tests\\data\\invalid_ext.webp", output_directory=out_dir_valid)