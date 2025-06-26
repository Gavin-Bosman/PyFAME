import pytest
from pyfame.manipulation.occlusion import LayerOcclusionPath
from pyfame.layer import Layer

def test_layer_occlusion_path():

    lop = LayerOcclusionPath()
    assert isinstance(lop, Layer) == True

    # testing fill_method checks
    with pytest.raises(TypeError):
        lop = LayerOcclusionPath(fill_method=2.0)
    with pytest.raises(ValueError):
        lop = LayerOcclusionPath(fill_method="n/a")