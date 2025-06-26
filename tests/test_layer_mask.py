import pytest
from pyfame.manipulation.mask import LayerMask
from pyfame.layer import Layer

def test_layer_mask():
    lm = LayerMask((0,0,0))
    assert isinstance(lm, Layer) == True
    
    # testing background_color checks
    with pytest.raises(TypeError):
        lm = LayerMask([0,0,0])
    with pytest.raises(TypeError):
        lm = LayerMask(("a", "b", "c"))
    with pytest.raises(ValueError):
        lm = LayerMask((1000, 432, 123))