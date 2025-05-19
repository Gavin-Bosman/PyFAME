import pytest
from pyfame.utils.exceptions import *
from pyfame.manipulations.occlusion.apply_occlusion import apply_noise

def test_exception_handling():
    in_dir_valid = "tests\\data\\sample_video.mp4"
    in_dir_invalid = "tests\\data\\Videos\\Actor_01.mp4"
    out_dir_valid = "tests\\data\\outputs"
    out_dir_invalid = "tests\\images\\masked_videos"

    # exception testing input_dir parameter
    with pytest.raises(TypeError):
        apply_noise(input_dir=1, output_dir=out_dir_valid)
    with pytest.raises(OSError):
        apply_noise(input_dir=in_dir_invalid, output_dir=out_dir_valid)
    
    # exception testing output_dir parameter
    with pytest.raises(TypeError):
        apply_noise(input_dir=in_dir_valid, output_dir=1)
    with pytest.raises(OSError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_invalid)
    with pytest.raises(ValueError):
        apply_noise(input_dir=in_dir_valid, output_dir=in_dir_valid)
    
    # exception testing noise_method parameter
    with pytest.raises(TypeError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, noise_method=2.0)
    with pytest.raises(ValueError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, noise_method=100)
    with pytest.raises(ValueError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, noise_method="test")
    
    # exception testing pixel_size parameter
    with pytest.raises(TypeError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, pixel_size="test")
    with pytest.raises(ValueError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, pixel_size=0)
    
    # exception testing noise_prob parameter
    with pytest.raises(TypeError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, noise_prob="test")
    with pytest.raises(ValueError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, noise_prob=1.5)
    
    # exception testing rand_seed parameter
    with pytest.raises(TypeError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, rand_seed="test")
    
    # exception testing mean parameter
    with pytest.raises(TypeError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, mean="test")
    
    # exception testing standard_dev parameter
    with pytest.raises(TypeError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, standard_dev="test")
    
    # exception testing mask_type parameter
    with pytest.raises(TypeError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, mask_type="test")
    with pytest.raises(ValueError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, mask_type=100)
    
    # exception testing with_sub_dirs parameter
    with pytest.raises(TypeError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, with_sub_dirs=1)
    
    # exception testing mediapipe configuration parameters
    with pytest.raises(TypeError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, min_detection_confidence=2)
    with pytest.raises(ValueError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, min_detection_confidence=2.2)
    
    with pytest.raises(TypeError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, min_tracking_confidence=2)
    with pytest.raises(ValueError):
        apply_noise(input_dir=in_dir_valid, output_dir=out_dir_valid, min_tracking_confidence=2.2)
    
    # Testing exceptions for input files with no face present
    with pytest.raises(FaceNotFoundError):
        apply_noise(input_dir="tests\\data\\no_face.png", output_dir=out_dir_valid)
    with pytest.raises(FaceNotFoundError):
        apply_noise(input_dir="tests\\data\\no_face.mp4", output_dir=out_dir_valid)
    
    # Testing exceptions for files encoded in an incompatible extension
    with pytest.raises(UnrecognizedExtensionError):
        apply_noise(input_dir="tests\\data\\invalid_ext.webp", output_dir=out_dir_valid)