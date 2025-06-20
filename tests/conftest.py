import os
import pytest
from pyfame.mesh import get_mesh

os.environ["PYTEST_RUNNING"] = "1"

@pytest.fixture
def valid_input_dir():
    in_dir_valid = "tests\\data\\"
    return in_dir_valid

@pytest.fixture
def invalid_input_dir():
    in_dir_invalid = "tests\\data\\Videos\\"
    return in_dir_invalid

@pytest.fixture
def valid_output_dir():
    out_dir_valid = "tests\\data\\outputs"
    return out_dir_valid

@pytest.fixture
def invalid_output_dir():
    out_dir_invalid = "tests\\images\\masked_videos"
    return out_dir_invalid

@pytest.fixture
def sample_video_path():
    file_path = "tests\\data\\sample_video.mp4"
    return file_path

@pytest.fixture
def sample_image_path():
    file_path = "tests\\data\\no_face.png"
    return file_path

@pytest.fixture
def face_mesh():
    return get_mesh(0.5, 0.5, False, 1)