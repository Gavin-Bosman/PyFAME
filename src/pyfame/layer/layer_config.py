from dataclasses import dataclass, field
from .layer import Layer
from typing import Callable, Any
from cv2.typing import MatLike
from pyfame.timing.timing_curves import timing_linear
from pyfame.util.util_checks import *
from pyfame.mesh.get_mesh_landmarks import FACE_OVAL_PATH

@dataclass
class LayerConfig:
    """ A configuration wrapper to the Layer class, allows each layer to have unique onset/offset times, 
    as well as unique timing functions.
    """
    layer: Layer
    onset_t: float
    offset_t: float
    timing_func: Callable[...,float]
    roi: list[list[tuple]]
    kwargs: dict[str,Any] = field(default_factory=dict)

    def __init__(self, layer:Layer, onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear,
                 roi:list[list[tuple]] = [FACE_OVAL_PATH], fade_duration:int = 500, **kwargs):
        check_type(layer, [Layer])
        check_type(onset_t, [float, None])
        check_type(offset_t, [float, None])
        check_type(timing_func, Callable)
        check_type(fade_duration, [int])

        self.layer = layer
        self.onset_t = onset_t
        self.offset_t = offset_t
        self.timing_func = timing_func
        self.roi = roi
        self.fade_duration = fade_duration/1000
        self.timing_kwargs = kwargs
    
    def compute_weight(self, dt:float) -> float:
        if dt < self.onset_t:
            return 0.0
        elif self.onset_t <= dt <= self.offset_t:
            return self.timing_func(dt, self.onset_t, self.offset_t, **self.timing_kwargs)
        elif self.offset_t <= dt <= self.offset_t + self.fade_duration:
            fade_t = (dt - self.offset_t)/self.fade_duration
            return self.timing_func(dt, self.onset_t, self.offset_t, **self.timing_kwargs) * (1-fade_t)
        else:
            return 0.0

    def apply_layer(self, frame:MatLike, dt:float=None) -> MatLike:
        if dt is not None:
            weight = self.compute_weight(dt)
            return self.layer.apply_layer(frame, self.roi, weight)
        else:
            return self.layer.apply_layer(frame, self.roi, 1.0)

