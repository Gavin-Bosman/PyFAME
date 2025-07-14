import pytest
from pyfame.manipulation.colour import LayerColor
from pyfame.layer import Layer

def test_layer_color():

    layer = LayerColor("red")
    assert isinstance(layer, Layer) == True
    
    # testing color checks
    with pytest.raises(TypeError):
        layer = LayerColor(focus_color=2.0)
    with pytest.raises(ValueError):
        layer = LayerColor(focus_color="purple")

    # testing magnitude checks
    with pytest.raises(TypeError): 
        layer = LayerColor("red", magnitude=2)
    with pytest.raises(ValueError):
        layer = LayerColor("red", magnitude=-5.0)