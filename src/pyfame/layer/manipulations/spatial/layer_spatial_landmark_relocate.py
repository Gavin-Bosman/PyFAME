from pydantic import BaseModel, ValidationError, PositiveInt
from typing import Optional
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.get_landmark_coordinates import get_pixel_coordinates_from_landmark
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations.mask import mask_from_landmarks
from pyfame.utilities.constants import *
import cv2 as cv
import numpy as np
from operator import itemgetter

class RelocateParameters(BaseModel):
    random_seed:Optional[PositiveInt]

class LayerSpatialLandmarkRelocate(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, relocation_parameters:RelocateParameters):
        self.time_config = timing_configuration
        self.relocate_params = relocation_parameters

        # Initialise the superclass
        super().__init__(self.time_config)       

        # Declare class parameters
        self.rand_seed = self.relocate_params.random_seed
        if self.rand_seed is None:
            self.rand_seed = np.random.randint(0,1000)

        # Snapshot of initial state
        self._snapshot_state()
    
    def supports_weight(self):
        return False

    def get_layer_parameters(self) -> dict:
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.relocate_params.model_dump())
        self._layer_parameters["time_onset"] = self.onset_t
        self._layer_parameters["time_offset"] = self.offset_t
        return dict(self._layer_parameters)
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float) -> cv.typing.MatLike:
        
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame

        # Create an rng instance 
        rng = np.random.default_rng(self.rand_seed)

        # The angle of rotation and x,y positons of the landmarks are randomly generated in the loop below
        rot_angles = {}
        x_displacements = {}

        for i in range(4):
            rn = rng.random()

            if i < 2:
                    if rn < 0.25:
                        rot_angles.update({i:90})
                    elif rn < 0.5:
                        rot_angles.update({i:-90})
                    elif rn < 0.75:
                        rot_angles.update({i:180})
                    else:
                        rot_angles.update({i:0})
            elif i == 2:
                if rn < 0.5:
                    rot_angles.update({i:90})
                else:
                    rot_angles.update({i:-90})
            else:
                if rn < 0.5:
                    rot_angles.update({i:180})
                else:
                    rot_angles.update({i:0})
            
            if rn < 0.5:
                x_displacements.update({i:int(-40 * rng.random())})
            else:
                x_displacements.update({i:int(40 * rng.random())})

        # Get the pixel coordinates of various landmark regions
        le_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_LEFT_EYE_REGION)
        re_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_RIGHT_EYE_REGION)
        nose_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_NOSE)
        lips_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_LIPS)

        # Creating boolean masks of each landmark region
        le_mask = mask_from_landmarks(frame, LANDMARK_LEFT_EYE_REGION, landmarker_coordinates)
        re_mask = mask_from_landmarks(frame, LANDMARK_RIGHT_EYE_REGION, landmarker_coordinates)
        nose_mask = mask_from_landmarks(frame, LANDMARK_NOSE, landmarker_coordinates)
        lip_mask = mask_from_landmarks(frame, LANDMARK_LIPS, landmarker_coordinates)
        fo_mask = mask_from_landmarks(frame, LANDMARK_FACE_OVAL, landmarker_coordinates)

        masks = [le_mask, re_mask, nose_mask, lip_mask]
        screen_coords = [le_screen_coords, re_screen_coords, nose_screen_coords, lips_screen_coords]
        lms = []
        output_frame = frame.copy()

        # Cut out, and store landmarks
        for mask, coords in zip(masks, screen_coords):
            im_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            im_mask[mask.astype(bool)] = 255

            max_x = max(coords, key=itemgetter(0))[0]
            min_x = min(coords, key=itemgetter(0))[0]

            max_y = max(coords, key=itemgetter(1))[1]
            min_y = min(coords, key=itemgetter(1))[1]

            # Compute the center bisecting lines of the landmark
            cy = int(round((max_y + min_y)/2))       
            cx = int(round((max_x + min_x)/2))

            # Cut out the current landmark region and store it
            lm = cv.bitwise_and(src1=output_frame, src2=output_frame, mask=im_mask)
            lms.append((lm, (cx,cy)))

            # fill the original lm region with empty pixels 
            output_frame = cv.bitwise_and(src1=output_frame, src2=output_frame, mask=cv.bitwise_not(im_mask))

            # Fill landmark holes with navier-stokes inpainting;
            # uses nearest-neighbor colour sampling to fill in gaps in an image
            output_frame = cv.inpaint(output_frame, im_mask, 15, cv.INPAINT_NS)

            # dilate the landmark mask slightly, and blur around inpainted edges
            kernel = np.ones((5,5), np.uint8)
            dilated_mask = cv.dilate(mask, kernel, iterations=1)
            dilated_mask = dilated_mask[..., np.newaxis]

            # blur the output frame around the inpainted landmarks
            face_only = cv.bitwise_and(output_frame, output_frame, mask=fo_mask)
            face_only = cv.GaussianBlur(face_only, (15,15), sigmaX=10)

            output_frame = np.where(dilated_mask == 255, face_only, output_frame)
        
        # Perform a weighted addition between the facial mean tone and the inpainted image
        facial_mean = cv.mean(frame, mask=fo_mask)[:3]
        fo_only = np.where(fo_mask[..., np.newaxis] == 255, facial_mean, frame).astype(np.uint8)
        output_frame = cv.addWeighted(output_frame, 0.6, fo_only, 0.4, 0)

        landmarks = dict(map(lambda i,j: (i,j), [0,1,2,3], lms))

        for key in list(landmarks.keys()):
            # Get the landmark, and the center point of its position
            landmark, center = landmarks[key]
            cx, cy = center
            h,w = landmark.shape[:2]

            # Get the current landmarks randomly generated rotation angle and x displacement
            rot_angle = rot_angles[key]
            x_disp = x_displacements[key]

            # Generate rotation matrices for the landmark
            if key == 2:
                rot_mat = cv.getRotationMatrix2D(center=center, angle=rot_angle, scale=1)
                landmark = cv.warpAffine(landmark, rot_mat, (w,h))
                cy += 20
            else:
                rot_mat = cv.getRotationMatrix2D(center=center, angle=rot_angle, scale=1)
                landmark = cv.warpAffine(landmark, rot_mat, (w,h))
            
            # Add the x displacement to the original center point to translate it
            cx += x_disp

            # Create landmark mask
            lm_mask = np.zeros((landmark.shape[0], landmark.shape[1]), dtype=np.uint8)
            lm_mask = np.where(landmark != 0, 255, 0)
            lm_mask = lm_mask.astype(np.uint8)
            
            # Clone the landmark onto the original face in its new position
            output_frame = cv.seamlessClone(landmark, output_frame, lm_mask, (cx, cy), cv.NORMAL_CLONE)

        return output_frame
        
def layer_spatial_landmark_relocate(timing_configuration:TimingConfiguration | None = None, random_seed:int | None = None) -> LayerSpatialLandmarkRelocate:
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = RelocateParameters(random_seed=random_seed)
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerSpatialLandmarkRelocate.__name__}: {e}")
    
    return LayerSpatialLandmarkRelocate(time_config, params)