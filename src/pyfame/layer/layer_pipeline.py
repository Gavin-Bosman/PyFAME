from pyfame.layer.layer_config import LayerConfig
from cv2.typing import MatLike

# Discuss with livingstone about passing just LayerConfig or both config and Layer
class LayerPipeline:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer:LayerConfig):
        self.layers.append(layer)
    
    def apply_layers(self, frame:MatLike, dt:float) -> MatLike:
        for layer in self.layers:
            frame = layer.apply_layer(frame, dt)
        return frame