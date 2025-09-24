from .layer import Layer
import cv2 as cv
from cv2.typing import MatLike
from pyfame.layer.manipulations.stylise.layer_stylise_optical_flow_dense import LayerStyliseOpticalFlowDense
from pyfame.layer.manipulations.stylise.layer_stylise_optical_flow_sparse import LayerStyliseOpticalFlowSparse
from typing import Any


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
        
        # Non-weighted manipulations order of precidence
        # Masking, blurring/noise, 
        
        weighted.extend(non_weighted)
        return weighted
    
    def apply_layers(self, face_mesh:Any, frame:MatLike, dt:float, **kwargs) -> MatLike:
        self.layers = self.enforce_layer_ordering()

        file_path = kwargs.get("file_path", None)

        for layer in self.layers:
            # Optical flow layers require the file path if a precomputed colour scale is specified
            if isinstance(layer, (LayerStyliseOpticalFlowDense, LayerStyliseOpticalFlowSparse)):
                frame = layer.apply_layer(face_mesh, frame, dt, file_path=file_path)
            else:
                frame = layer.apply_layer(face_mesh, frame, dt)
        return frame