from .layer_config import LayerConfig
from cv2.typing import MatLike

# Discuss with livingstone about passing just LayerConfig or both config and Layer
class LayerPipeline:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer:LayerConfig):
        self.layers.append(layer)
    
    def enforce_layer_ordering(self) -> list[LayerConfig]:
        # put non weighted manipulations like masking and occlusion
        # at the end, to not disrupt the full-frame manipulations
        weighted = []
        non_weighted = []

        for config in self.layers:
            if config.layer.supports_weight():
                weighted.append(config)
            else:
                non_weighted.append(config)
        
        weighted.extend(non_weighted)
        return weighted
    
    def apply_layers(self, frame:MatLike, dt:float) -> MatLike:
        self.layers = self.enforce_layer_ordering()

        for layer in self.layers:
            frame = layer.apply_layer(frame, dt)
        return frame