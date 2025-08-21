from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, NonNegativeFloat, PositiveInt
from typing import Union, List, Tuple, Optional
from pyfame.mesh import *
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_path
from pyfame.utilities.constants import *
import cv2 as cv
import numpy as np
from skimage.util import *

class NoiseParameters(BaseModel):
    random_seed:Optional[int]
    noise_method:Union[int,str]
    noise_probability:NonNegativeFloat
    pixel_size:PositiveInt
    gaussian_mean:float
    gaussian_deviation:NonNegativeFloat
    region_of_interest:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]

    @field_validator("noise_method", mode="before")
    @classmethod
    def check_compatible_value(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if isinstance(value, str):
            value = str.lower(value)
            if value not in {"pixelate", "salt and pepper", "gaussian"}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        elif isinstance(value, int):
            if value not in {18,19,20}:
                raise ValueError(f"Unrecognized value for parameter {field_name}.")
            return value
        
        raise TypeError(f"{field_name} provided an invalid type. Must be one of int, str.")
    
    @field_validator("pixel_size")
    @classmethod
    def check_compatible_size(cls, value, info:ValidationInfo):
        field_name = info.field_name

        if value < 4:
            raise ValueError(f"{field_name} requires a size >= 4.")

class LayerOcclusionNoise(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, noise_parameters:NoiseParameters):

        self.time_config = timing_configuration
        self.noise_params = noise_parameters

        # Initialising superclass
        super().__init__(self.time_config)

        # Defining class parameters
        self.rand_seed = self.noise_params.random_seed
        self.noise_method = self.noise_params.noise_method
        self.noise_probability = self.noise_params.noise_probability
        self.pixel_size = self.noise_params.pixel_size
        self.mean = self.noise_params.gaussian_mean
        self.standard_deviation = self.noise_params.gaussian_deviation
        self.region_of_interest = self.noise_params.region_of_interest
        self.min_tracking_confidence = self.time_config.min_tracking_confidence
        self.min_detection_confidence = self.time_config.min_detection_confidence
        self.static_image_mode = False

        # Dump pydantic models to get full param list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.noise_params.model_dump())
    
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

def layer_occlusion_noise(timing_configuration:TimingConfiguration | None = None, region_of_interest:list[list[tuple[int,...]]] | list[tuple[int,...]] = FACE_OVAL_PATH, noise_method:int|str = "gaussian", 
                          noise_probability:float = 0.5, pixel_size:int = 32, mean:float = 0.0, standard_deviation:float = 0.5, random_seed:int|None = None) -> LayerOcclusionNoise:
    
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = NoiseParameters(
            random_seed=random_seed, 
            noise_method=noise_method, 
            noise_probability=noise_probability, 
            pixel_size=pixel_size, 
            gaussian_mean=mean, 
            gaussian_deviation=standard_deviation, 
            region_of_interest=region_of_interest
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerOcclusionNoise.__name__}: {e}")
    
    return LayerOcclusionNoise(time_config, params)