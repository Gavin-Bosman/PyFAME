from pyfame.util.util_constants import *
from pyfame.mesh import *
from pyfame.util.util_checks import *
from pyfame.layer import Layer
from pyfame.timing.timing_curves import timing_linear
import cv2 as cv
import numpy as np
from skimage.util import *
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerOcclusionNoise(Layer):
    def __init__(self, rand_seed:int|None, method:int|str = "gaussian", noise_prob:float = 0.5, pixel_size:int = 32, mean:float = 0.0, standard_dev:float = 0.5, 
                 onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, roi:list[list[tuple]] = [FACE_OVAL_PATH], 
                 fade_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        super().__init__(onset_t, offset_t, timing_func, roi, fade_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        check_type(method, [int, str])
        check_value(method, [18,19,20,"pixelate","salt and pepper","gaussian"])
        check_type(noise_prob, [float])
        check_value(noise_prob, min=0.0, max=1.0)
        check_type(pixel_size, [int])
        check_value(pixel_size, min=4)
        check_type(mean, [float])
        check_type(standard_dev, [float])

        self.rand_seed = rand_seed
        self.method = method
        self.prob = noise_prob
        self.pixel_size = pixel_size
        self.mean = mean
        self.sd = standard_dev
        self.roi = roi
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.static_image_mode = False
    
    def supports_weight(self):
        return False

    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):

        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            rng = None
            if self.rand_seed is not None:
                rng = np.random.default_rng(self.rand_seed)
            else:
                rng = np.random.default_rng()

            face_mesh = super().get_face_mesh()
            mask = get_mask_from_path(frame, self.roi, face_mesh)
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
            output_frame = frame.copy()

            match self.method:
                case "pixelate" | 18:
                    height, width = frame.shape[:2]
                    h = frame.shape[0]//self.pixel_size
                    w = frame.shape[1]//self.pixel_size

                    temp = cv.resize(frame, (w, h), None, 0, 0, cv.INTER_LINEAR)
                    output_frame = cv.resize(temp, (width, height), None, 0, 0, cv.INTER_NEAREST)

                    output_frame = np.where(mask == 255, output_frame, frame)
                
                case "salt and pepper" | 19:
                    # Divide prob in 2 for "salt" and "pepper"
                    thresh = self.prob
                    noise_prob = self.prob/2
                    
                    # Use numpy's random number generator to generate a random matrix in the shape of the frame
                    rdm = rng.random(frame.shape[:2])

                    # Create boolean masks 
                    pepper_mask = rdm < noise_prob
                    salt_mask = (rdm >= noise_prob) & (rdm < thresh)
                    
                    # Apply boolean masks
                    output_frame[pepper_mask] = [0,0,0]
                    output_frame[salt_mask] = [255,255,255]

                    output_frame = np.where(mask == 255, output_frame, frame)
                
                case "gaussian" | 20:
                    var = self.sd**2

                    # scikit-image's random_noise function works with floating point images, need to convert our frame's type
                    output_frame = img_as_float64(output_frame)
                    output_frame = random_noise(image=output_frame, mode='gaussian', rng=rng, mean=self.mean, var=var)
                    output_frame = img_as_ubyte(output_frame)

                    output_frame = np.where(mask == 255, output_frame, frame)
            
            return output_frame