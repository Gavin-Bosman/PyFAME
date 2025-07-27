from pyfame.utilities.constants import *
from pyfame.mesh import *
from pyfame.utilities.checks import *
from pyfame.utilities.general_utilities import sanitize_json_value, get_roi_name
from pyfame.layer.layer import Layer
from pyfame.layer.timing_curves import timing_linear
from pyfame.layer.manipulations.mask import mask_from_path
import cv2 as cv
import numpy as np
from skimage.util import *
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class LayerOcclusionNoise(Layer):
    def __init__(self, rand_seed:int|None, method:int|str = "gaussian", noise_prob:float = 0.5, pixel_size:int = 32, mean:float = 0.0, standard_dev:float = 0.5, 
                 onset_t:float=None, offset_t:float=None, timing_func:Callable[...,float]=timing_linear, roi:list[list[tuple]] | list[tuple] = FACE_OVAL_PATH, 
                 rise_duration:int=500, fall_duration:int = 500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs):
        # Initialising superclass
        super().__init__(onset_t, offset_t, timing_func, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)
        self.static_image_mode = False
        self._pre_attrs = []
        self._pre_attrs = set(self.__dict__) # snapshot of just the superclass parameters
        
        # Performing input parameter checks
        check_type(method, [int, str])
        check_value(method, [18,19,20,"pixelate","salt and pepper","gaussian"])
        check_type(noise_prob, [float])
        check_value(noise_prob, min=0.0, max=1.0)
        check_type(pixel_size, [int])
        check_value(pixel_size, min=4)
        check_type(mean, [float])
        check_type(standard_dev, [float])

        # Defining class parameters
        self.rand_seed = rand_seed
        self.noise_method = method
        self.noise_probability = noise_prob
        self.pixel_size = pixel_size
        self.mean = mean
        self.standard_deviation = standard_dev
        self.time_onset = onset_t
        self.time_offset = offset_t
        self.timing_function = timing_func
        self.region_of_interest = roi
        self.rise_duration_msec = rise_duration
        self.fall_duration_msec = fall_duration
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.timing_kwargs = kwargs

        self._capture_init_params()
    
    def _capture_init_params(self):
        # Extracting total parameter list post init
        post_attrs = set(self.__dict__.keys())

        # Getting only the subclass parameters
        new_attrs = post_attrs - self._pre_attrs

        # Store only subclass level params; ignore self
        params = {attr: getattr(self, attr) for attr in new_attrs}

        # Handle non serializable types
        if "region_of_interest" in params:
            params["region_of_interest"] = get_roi_name(params["region_of_interest"])

        self._layer_parameters = {
            k: sanitize_json_value(v) for k, v in params.items()
        }
    
    def supports_weight(self):
        return False

    def get_layer_parameters(self):
        return dict(self._layer_parameters)

    def apply_layer(self, frame:cv.typing.MatLike, dt:float = None, static_image_mode:bool = False):

        # Update the faceMesh when switching between image and video processing
        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        # This layer does not support weight; weight will always be 0.0 or 1.0
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            # Create an rng instance to help generate random noise
            rng = None
            if self.rand_seed is not None:
                rng = np.random.default_rng(self.rand_seed)
            else:
                rng = np.random.default_rng()

            face_mesh = super().get_face_mesh()
            mask = mask_from_path(frame, self.region_of_interest, face_mesh)
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
            output_frame = frame.copy()

            match self.noise_method:
                case "pixelate" | 18:
                    height, width = frame.shape[:2]
                    h = frame.shape[0]//self.pixel_size
                    w = frame.shape[1]//self.pixel_size

                    # resizing the pixels of the image in the region of interest
                    temp = cv.resize(frame, (w, h), None, 0, 0, cv.INTER_LINEAR)
                    output_frame = cv.resize(temp, (width, height), None, 0, 0, cv.INTER_NEAREST)

                    output_frame = np.where(mask == 255, output_frame, frame)
                
                case "salt and pepper" | 19:
                    # Divide prob in 2 for "salt" and "pepper"
                    thresh = self.noise_probability
                    noise_prob = self.noise_probability/2
                    
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
                    var = self.standard_deviation**2

                    # scikit-image's random_noise function works with floating point images; we need to pre-convert our frames to float64
                    output_frame = img_as_float64(output_frame)
                    output_frame = random_noise(image=output_frame, mode='gaussian', rng=rng, mean=self.mean, var=var)
                    output_frame = img_as_ubyte(output_frame)

                    output_frame = np.where(mask == 255, output_frame, frame)
            
            return output_frame

def layer_occlusion_noise(rand_seed:int|None, method:int|str = "gaussian", noise_probability:float = 0.5, pixel_size:int = 32, mean:float = 0.0, standard_deviation:float = 0.5, 
                         time_onset:float=None, time_offset:float=None, timing_function:Callable[...,float]=timing_linear, region_of_interest:list[list[tuple]] | list[tuple] = FACE_OVAL_PATH, 
                         rise_duration:int=500, fall_duration:int=500, min_tracking_confidence:float = 0.5, min_detection_confidence:float = 0.5, **kwargs) -> LayerOcclusionNoise:
    
    return LayerOcclusionNoise(rand_seed, method, noise_probability, pixel_size, mean, standard_deviation, time_onset, time_offset,
                               timing_function, region_of_interest, rise_duration, fall_duration, min_tracking_confidence, min_detection_confidence, **kwargs)