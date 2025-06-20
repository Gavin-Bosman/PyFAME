from pyfame.util.util_constants import *
from pyfame.mesh import *
from pyfame.util.util_exceptions import *
from pyfame.layer import Layer
import cv2 as cv
import mediapipe as mp
import numpy as np
from skimage.util import *
import logging

logger = logging.getLogger("pyfame")
debug_logger = logging.getLogger("pyfame.debug")

class layer_occlusion_blur(Layer):
    def __init__(self, face_mesh:mp.solutions.face_mesh.FaceMesh, method:str|int = "gaussian", kernel_size:int = 15):
        self.face_mesh = face_mesh
        self.method = method
        self.k_size = kernel_size

        if isinstance(method, str):
            self.method = str.lower(method)
    
    def apply_layer(self, frame, weight, roi):
        mask = get_mask_from_path(frame, roi, self.face_mesh)
        output_frame = np.zeros_like(frame, dtype=np.uint8)

        match self.method:
            case "average" | 11:
                frame_blurred = cv.blur(frame, (self.k_size, self.k_size))
                output_frame = np.where(mask == 255, frame_blurred, frame)
            
            case "gaussian" | 12:
                frame_blurred = cv.GaussianBlur(frame, (self.k_size, self.k_size), 0)
                output_frame = np.where(mask == 255, frame_blurred, frame)
            
            case "median" | 13:
                frame_blurred = cv.medianBlur(frame, self.k_size)
                output_frame = np.where(mask == 255, frame_blurred, frame)
        
        return output_frame

class layer_occlusion_noise(Layer):
    def __init__(self, face_mesh:mp.solutions.face_mesh.FaceMesh, rand_seed:int|None, method:int|str = "gaussian", 
                 noise_prob:float = 0.5, pixel_size:int = 32, mean:float = 0.0, standard_dev:float = 0.5):
        self.face_mesh = face_mesh
        self.rand_seed = rand_seed

        if isinstance(method, str):
            self.method = str.lower(method)
        else:
            self.method = method
        
        self.prob = noise_prob
        self.pixel_size = pixel_size
        self.mean = mean
        self.sd = standard_dev

    def apply_layer(self, frame, weight, roi):
        rng = None
        if self.rand_seed is not None:
            rng = np.random.default_rng(self.rand_seed)
        else:
            rng = np.random.default_rng()

        mask = get_mask_from_path(frame, roi, self.face_mesh)
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