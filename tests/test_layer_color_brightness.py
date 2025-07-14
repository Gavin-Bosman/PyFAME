import pytest
from pyfame.manipulation.colour import LayerColorBrightness
from pyfame.layer import Layer

def test_layer_color_brightness():
    lb = LayerColorBrightness(20.0)
    assert isinstance(lb, Layer) == True
    
    # testing magnitude checks
    with pytest.raises(TypeError):
        lb = LayerColorBrightness("na")
    with pytest.raises(ValueError):
        lb = LayerColorBrightness(-2.0)