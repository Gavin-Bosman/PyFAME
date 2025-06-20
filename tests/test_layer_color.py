import pytest
from pyfame.manipulation.color import layer_color
from pyfame.layer import Layer

def test_layer_color(face_mesh):

    layer = layer_color(face_mesh, "red")
    assert isinstance(layer, Layer) == True

    with pytest.raises(ValueError):
        layer = layer_color(12, "red")
    with pytest.raises(TypeError):
        layer = layer_color(face_mesh, color=2.0)
    with pytest.raises(ValueError):
        layer = layer_color(face_mesh, color="purple")
    with pytest.raises(TypeError): 
        layer = layer_color(face_mesh, "red", magnitude=2)
    with pytest.raises(ValueError):
        layer = layer_color(face_mesh, "red", magnitude=-5.0)
    with pytest.raises(TypeError):
        layer = layer_color(face_mesh, "red", timing_func=3.2)
    with pytest.raises(TypeError):
        layer = layer_color(face_mesh, "red", timing_func=locals())