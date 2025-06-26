import pytest
from pyfame.manipulation.occlusion import LayerOcclusionBar
from pyfame.layer import Layer

def test_layer_occlusion_bar():
    lob = LayerOcclusionBar()
    assert isinstance(lob, Layer) == True

    # testing background_color checks
    with pytest.raises(TypeError):
        lob = LayerOcclusionBar([1,2,3])
    with pytest.raises(TypeError):
        lob = LayerOcclusionBar(("a", "b", "c"))