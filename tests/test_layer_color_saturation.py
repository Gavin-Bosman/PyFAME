import pytest
from pyfame.manipulation.colour import LayerColorSaturation
from pyfame.layer import Layer

def test_layer_color_saturation(face_mesh):
    ls = LayerColorSaturation(10.0)
    assert isinstance(ls, Layer) == True
    
    # testing magnitude checks
    with pytest.raises(TypeError):
        ls = LayerColorSaturation("na")
    with pytest.raises(ValueError):
        ls = LayerColorSaturation(-2.0)