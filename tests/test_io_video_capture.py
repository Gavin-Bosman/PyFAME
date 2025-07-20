import pytest
import cv2 as cv
from pyfame.file_access import get_video_capture
from pyfame.utilities import exceptions

def test_get_video_capture(valid_input_dir, invalid_input_dir, sample_video_path, sample_image_path):

    # Type checking return value
    vc = get_video_capture(sample_video_path)
    assert isinstance(vc, cv.VideoCapture) == True

    # Checking parameter error handling
    with pytest.raises(TypeError):
        get_video_capture(1)
    with pytest.raises(OSError):
        get_video_capture(invalid_input_dir)
    with pytest.raises(ValueError):
        get_video_capture(valid_input_dir)
    with pytest.raises(exceptions.UnrecognizedExtensionError):
        get_video_capture(sample_image_path)