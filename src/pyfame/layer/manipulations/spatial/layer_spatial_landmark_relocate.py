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
        else:
            # Create an rng instance 
            rng = None
            if self.rand_seed is not None:
                rng = np.random.default_rng(self.rand_seed)
            else:
                rng = np.random.default_rng()

            fo_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_FACE_OVAL)
            
            # Get x and y bounds of the face oval
            max_x = max(fo_screen_coords, key=itemgetter(0))[0]
            min_x = min(fo_screen_coords, key=itemgetter(0))[0]

            max_y = max(fo_screen_coords, key=itemgetter(1))[1]
            min_y = min(fo_screen_coords, key=itemgetter(1))[1]

            # The angle of rotation and x,y positons of the landmarks are randomly generated in the loop below
            rot_angles = {}
            x_displacements = {}

            for i in range(4):
                rn = rng.random()

                if i+1 < 3:
                        if rn < 0.25:
                            rot_angles.update({i+1:90})
                        elif rn < 0.5:
                            rot_angles.update({i+1:-90})
                        elif rn < 0.75:
                            rot_angles.update({i+1:180})
                        else:
                            rot_angles.update({i+1:0})
                elif i+1 == 3:
                    if rn < 0.5:
                        rot_angles.update({i+1:90})
                    else:
                        rot_angles.update({i+1:-90})
                else:
                    if rn < 0.5:
                        rot_angles.update({i+1:180})
                    else:
                        rot_angles.update({i+1:0})
                
                if rn < 0.5:
                    x_displacements.update({i+1:int(-40 * rng.random())})
                else:
                    x_displacements.update({i+1:int(40 * rng.random())})

            # Get the pixel coordinates of various landmark regions
            le_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_LEFT_EYE_REGION)
            re_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_RIGHT_EYE_REGION)
            nose_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_NOSE)
            lips_screen_coords = get_pixel_coordinates_from_landmark(landmarker_coordinates, LANDMARK_MOUTH_REGION)

            # Creating boolean masks of each landmark region
            le_mask = mask_from_landmarks(frame, LANDMARK_LEFT_EYE_REGION, landmarker_coordinates).astype(bool)
            re_mask = mask_from_landmarks(frame, LANDMARK_RIGHT_EYE_REGION, landmarker_coordinates).astype(bool)
            nose_mask = mask_from_landmarks(frame, LANDMARK_NOSE, landmarker_coordinates).astype(bool)
            lip_mask = mask_from_landmarks(frame, LANDMARK_MOUTH_REGION, landmarker_coordinates).astype(bool)
            fo_mask = mask_from_landmarks(frame, LANDMARK_FACE_OVAL, landmarker_coordinates).astype(bool)

            masks = [le_mask, re_mask, nose_mask, lip_mask]
            screen_coords = [le_screen_coords, re_screen_coords, nose_screen_coords, lips_screen_coords]
            lms = []
            output_frame = frame.copy()

            # Cut out, and store landmarks
            for mask, coords in zip(masks, screen_coords):
                im_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                im_mask[mask] = 255

                max_x = max(coords, key=itemgetter(0))[0]
                min_x = min(coords, key=itemgetter(0))[0]

                max_y = max(coords, key=itemgetter(1))[1]
                min_y = min(coords, key=itemgetter(1))[1]

                # Compute the center bisecting lines of the landmark
                cx = round((max_y + min_y)/2)           
                cy = round((max_x + min_x)/2)

                # Cut out the current landmark region and store it
                lm = cv.bitwise_and(src1=frame, src2=frame, mask=im_mask)
                lms.append((lm, (cy,cx)))

                # Fill landmark holes with navier-stokes inpainting;
                # uses nearest-neighbor colour sampling to fill in gaps in an image
                output_frame[mask] = 0
                output_frame = cv.inpaint(output_frame, im_mask, 10, cv.INPAINT_NS)

            landmarks = dict(map(lambda i,j: (i,j), [1,2,3,4], lms))

            for key in landmarks:
                # Get the landmark, and the center point of its position
                landmark, center = landmarks[key]
                cx, cy = center
                h,w = landmark.shape[:2]

                # Get the current landmarks randomly generated rotation angle and x displacement
                rot_angle = rot_angles[key]
                x_disp = x_displacements[key]

                # Generate rotation matrices for the landmark
                if key == 3:
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