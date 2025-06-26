import pytest 
from pyfame.manipulation.occlusion import LayerOcclusionNoise
from pyfame.layer import Layer

def test_layer_occlusion_noise():
    lon = LayerOcclusionNoise(1234, "gaussian")
    assert isinstance(lon, Layer) == True

    # testing method checks
    with pytest.raises(TypeError):
        lon = LayerOcclusionNoise(1234, method=2.0)
    with pytest.raises(ValueError):
        lon = LayerOcclusionNoise(1234, method="n/a")
    
    # testing noise_prob checks
    with pytest.raises(TypeError):
        lon = LayerOcclusionNoise(1234, "gaussian", "na")
    with pytest.raises(ValueError):
        lon = LayerOcclusionNoise(1234, "gaussian", 10.0)
    
    # testing pixel_size checks
    with pytest.raises(TypeError):
        lon = LayerOcclusionNoise(pixel_size=2.0)
    with pytest.raises(ValueError):
        lon = LayerOcclusionNoise(pixel_size=1)
    
    # testing mean checks
    with pytest.raises(TypeError):
        lon = LayerOcclusionNoise(mean="n/a")
    
    # testing sd checks
    with pytest.raises(TypeError):
        lon = LayerOcclusionNoise(standard_dev="n/a")