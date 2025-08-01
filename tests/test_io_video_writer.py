import pytest
import cv2 as cv
from pyfame.file_access import get_video_writer

def test_get_video_writer(valid_input_dir, invalid_input_dir, sample_video_path):
    valid_size = (500,700)

    vw = get_video_writer(valid_input_dir, valid_size)
    assert isinstance(vw, cv.VideoWriter) == True

    with pytest.raises(TypeError):
        get_video_writer(1, valid_size)
    with pytest.raises(OSError):
        get_video_writer(invalid_input_dir, valid_size)
    with pytest.raises(ValueError):
        get_video_writer(sample_video_path, valid_size)
    
    with pytest.raises(TypeError):
        get_video_writer(valid_input_dir, frame_size=[12,34])
    with pytest.raises(TypeError):
        get_video_writer(valid_input_dir, frame_size=(2.5, True))
    
    with pytest.raises(TypeError):
        get_video_writer(valid_input_dir, valid_size, video_codec=1)
    with pytest.raises(TypeError):
        get_video_writer(valid_input_dir, valid_size, frame_rate="none")
    with pytest.raises(TypeError):
        get_video_writer(valid_input_dir, valid_size, isColor="none")