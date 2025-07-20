import pytest
from pyfame.utilities.exceptions import *
from pyfame.manipulation.temporal.apply_temporal_frame_shuffle import generate_shuffled_block_array

def test_exception_handling():
    in_dir_valid = "tests\\data\\sample_video.mp4"
    in_dir_invalid = "tests\\data\\Videos\\Actor_01.mp4"
    out_dir_valid = "tests\\data\\outputs"
    
    # exception testing file_path parameter
    with pytest.raises(TypeError):
        generate_shuffled_block_array(file_path=1)
    with pytest.raises(OSError):
        generate_shuffled_block_array(file_path=in_dir_invalid)
    with pytest.raises(ValueError):
        generate_shuffled_block_array(file_path=out_dir_valid)
    
    # exception testing shuffle_method parameter
    with pytest.raises(TypeError):
        generate_shuffled_block_array(file_path=in_dir_valid, shuffle_method="test")
    with pytest.raises(ValueError):
        generate_shuffled_block_array(file_path=in_dir_valid, shuffle_method=100)
    
    # exception testing rand_seed parameter
    with pytest.raises(TypeError):
        generate_shuffled_block_array(file_path=in_dir_valid, rand_seed=2.5)
    
    # exception testing block_duration parameter
    with pytest.raises(TypeError):
        generate_shuffled_block_array(file_path=in_dir_valid, block_duration="test")
    with pytest.raises(ValueError):
        generate_shuffled_block_array(file_path=in_dir_valid, block_duration=30)