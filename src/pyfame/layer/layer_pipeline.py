from .layer import Layer
from cv2.typing import MatLike

class LayerPipeline:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer:Layer):
        self.layers.append(layer)
    
    def add_layers(self, layers:list[Layer]):
        if not self.layers:
            self.layers = layers
        else:
            self.layers.extend(layers)
    
    def enforce_layer_ordering(self) -> list[Layer]:
        # put non weighted manipulations like masking and occlusion
        # at the end, to not disrupt the full-frame manipulations
        weighted = []
        non_weighted = []

        for layer in self.layers:
            if layer.supports_weight():
                weighted.append(layer)
            else:
                non_weighted.append(layer)
        
        weighted.extend(non_weighted)
        return weighted
    
    def apply_layers(self, frame:MatLike, dt:float, static_image_mode:bool = False) -> MatLike:
        self.layers = self.enforce_layer_ordering()

        for layer in self.layers:
            frame = layer.apply_layer(frame, dt, static_image_mode)
        return frame