from pyfame.layer.layer import layer
from cv2.typing import MatLike

class layer_pipeline:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer:layer):
        self.layers.append(layer)
    
    def apply_layers(self, frame:MatLike, weight:float, roi:list[list[tuple]]) -> MatLike:
        for layer in self.layers:
            frame = layer.apply_layer(frame, weight, roi)
        return frame