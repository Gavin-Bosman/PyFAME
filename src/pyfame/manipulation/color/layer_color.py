from pyfame.util.util_constants import *
from pyfame.mesh import get_mask_from_path
from pyfame.layer import Layer
from pyfame.util.util_exceptions import *
from pyfame.timing.timing_curves import timing_constant
from pyfame.util.util_checks import *
import cv2 as cv
import mediapipe as mp
import numpy as np
import logging
from typing import Callable

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class layer_color(Layer):
    def __init__(self, face_mesh:mp.solutions.face_mesh.FaceMesh, color:str|int, magnitude:float = 10.0, 
                 timing_func:Callable[..., float] = timing_constant, **timing_kwargs):
        
        check_has_callable_attr(face_mesh, 'process')
        check_type(color, [str,int])
        check_value(color, ["red", "green", "blue", "yellow", 4, 5, 6, 7])
        check_type(magnitude, [float])
        check_value(magnitude, min=0)
        check_type(timing_func, Callable)
        check_return_type(timing_func, 0.0, [float])

        self.face_mesh = face_mesh
        self.magnitude = magnitude
        self.color = color
        self.timing_func = timing_func
        self.timing_kwargs = timing_kwargs
    
    def __get_timing_weight(self, dt:float) -> float:
        return self.timing_func(dt, **self.timing_kwargs)
    
    def apply_layer(self, frame:cv.typing.MatLike, roi:list[list[tuple]], dt:float):
        mask = get_mask_from_path(frame, roi, self.face_mesh)
        weight = self.__get_timing_weight(dt)

        # Convert input image to CIE La*b* color space (perceptually uniform space)
        img_LAB = cv.cvtColor(frame, cv.COLOR_BGR2LAB).astype(np.float32)
        l,a,b = cv.split(img_LAB)

        match self.color:
            case "red" | 4:
                a = np.where(mask==255, a + (weight * self.magnitude), a)
                np.clip(a, -128, 127)
            case "blue" | 5:
                b = np.where(mask==255, b - (weight * self.magnitude), b)
                np.clip(b, -128, 127)
            case "green" | 6:
                a = np.where(mask==255, a - (weight * self.magnitude), a)
                np.clip(a, -128, 127)
            case "yellow" | 7:
                b = np.where(mask==255, b + (weight * self.magnitude), b)
                np.clip(b, -128, 127)
        
        img_LAB = cv.merge([l,a,b])

        # Convert CIE La*b* back to BGR
        result = cv.cvtColor(img_LAB.astype(np.uint8), cv.COLOR_LAB2BGR)
        return result