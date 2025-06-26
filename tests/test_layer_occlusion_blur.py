import pytest
from pyfame.manipulation.occlusion import LayerOcclusionBlur
from pyfame.layer import Layer

def test_layer_occlusion_blur():
    lob = LayerOcclusionBlur("gaussian", 13)
    assert isinstance(lob, Layer) == True

    # testing method checks
    with pytest.raises(TypeError):
        lob = LayerOcclusionBlur(1.0)
    with pytest.raises(ValueError):
        lob = LayerOcclusionBlur("n/a")
    
    # tesing kernel_size checks
    with pytest.raises(TypeError):
        lob = LayerOcclusionBlur(kernel_size="na")
    with pytest.raises(ValueError):
        lob = LayerOcclusionBlur(kernel_size=0)