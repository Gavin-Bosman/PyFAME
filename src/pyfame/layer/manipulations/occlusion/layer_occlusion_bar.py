from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo, NonNegativeInt
from typing import Tuple, List
from pyfame.utilities.general_utilities import compute_rotation_angle, compute_slope
from pyfame.utilities.constants import *
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.landmark.facial_landmarks import *
from pyfame.landmark.get_landmark_coordinates import get_pixel_coordinates_from_landmark
import cv2 as cv
import numpy as np

class BarOcclusionParameters(BaseModel):
    bar_colour:Tuple[int,int,int]
    bar_padding:NonNegativeInt
    landmark_path:List[Tuple[int,...]]

    @field_validator("bar_colour")
    @classmethod
    def check_in_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        for elem in value:
            if not (0 <= elem <= 255):
                raise ValueError(f"{field_name} values must lie between 0 and 255.")
        
        return value
    
    @field_validator("landmark_path", mode="before")
    @classmethod
    def check_compatible_path(cls, value, info:ValidationInfo):
        valid_paths = [LANDMARK_LEFT_EYE_REGION, LANDMARK_RIGHT_EYE_REGION, LANDMARK_BOTH_EYE_REGIONS, LANDMARK_NOSE, LANDMARK_LIPS, LANDMARK_MOUTH_REGION]
        field_name = info.field_name
                
        if value not in valid_paths:
            raise ValueError(f"Incompatible path provided in {field_name}. Please provide one of: LEFT_EYE_PATH, RIGHT_EYE_PATH, BOTH_EYES_PATH, NOSE_PATH, LIPS_PATH, MOUTH_PATH.")
        
        return value

class LayerOcclusionBar(Layer):
    def __init__(self, timing_configuration:TimingConfiguration, occlusion_parameters:BarOcclusionParameters):

        self.time_config = timing_configuration
        self.occlude_params = occlusion_parameters

        # Initialise superclass
        super().__init__(self.time_config)

        # Define class parameters
        self.bar_color = self.occlude_params.bar_colour
        self.bar_padding = self.occlude_params.bar_padding
        self.landmark_path = self.occlude_params.landmark_path
        self.min_x_lm_id = -1
        self.max_x_lm_id = -1
        self.min_y_lm_id = -1
        self.max_y_lm_id = -1

        # Snapshot of initial state
        self._snapshot_state()
            
    def supports_weight(self):
        return False
    
    def get_layer_parameters(self) -> dict:
        # Dump the pydantic models to get dict of full parameter list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.occlude_params.model_dump())
        self._layer_parameters["time_onset"] = self.onset_t
        self._layer_parameters["time_offset"] = self.offset_t
        return dict(self._layer_parameters)
    
    def set_min_max_landmarks(self, landmarker_coordinates, coordinate_array):
        # Determine the landmark id's containing the min/max x-values
            xs = coordinate_array[:, 0]
            ys = coordinate_array[:, 1]
            min_x = int(xs.min())
            max_x = int(xs.max())
            min_y = int(ys.min())
            max_y = int(ys.max())
            
            min_x_candidates = [(i, lm) for i,lm in enumerate(landmarker_coordinates) if lm[0] == min_x]
            max_x_candidates = [(i, lm) for i,lm in enumerate(landmarker_coordinates) if lm[0] == max_x]
            min_y_candidates = [(i, lm) for i,lm in enumerate(landmarker_coordinates) if lm[1] == min_y]
            max_y_candidates = [(i, lm) for i,lm in enumerate(landmarker_coordinates) if lm[1] == max_y]

            best_pair_x = None
            best_pair_y = None
            min_vert_diff = float("inf")
            max_horiz_diff = 0

            for i1, lm1 in min_x_candidates:
                for i2, lm2 in max_x_candidates:
                    dy = abs(lm2[1] - lm1[1])
                    if dy < min_vert_diff:
                        min_vert_diff = dy
                        best_pair_x = (i1, i2)
            
            for i1, lm1 in min_y_candidates:
                for i2, lm2 in max_y_candidates:
                    dx = abs(lm2[0] - lm1[0])
                    if dx > max_horiz_diff:
                        max_horiz_diff = dx
                        best_pair_y = (i1, i2)
            
            self.min_x_lm_id, self.max_x_lm_id = best_pair_x
            self.min_y_lm_id, self.max_y_lm_id = best_pair_y
    
    def apply_layer(self, landmarker_coordinates:list[tuple[int,int]], frame:cv.typing.MatLike, dt:float):

        # Bar occlusion does not support weight, so weight will always be 0.0 or 1.0
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            h,w = frame.shape[:2]
            # Replace placeholder concave path with its convex sub-paths
            roi_coordinates = get_pixel_coordinates_from_landmark(landmarker_coordinates, self.landmark_path)
            roi_arr = np.array(roi_coordinates, dtype=int)

            if self.min_x_lm_id == -1 or self.max_x_lm_id == -1 or self.min_y_lm_id == -1 or self.max_y_lm_id == -1:
                self.set_min_max_landmarks(landmarker_coordinates, roi_arr)

            # Calculate the slope of the connecting line & angle to the horizontal
            # landmarks 162, 389 form a paralell line to the x-axis when the face is vertical
            p1 = landmarker_coordinates[162]
            p2 = landmarker_coordinates[389]
            slope = compute_slope(p1, p2)
            rot_angle = compute_rotation_angle(slope_1=slope)
            
            # Compute the center bisecting line of the landmark
            min_x_lm = landmarker_coordinates[self.min_x_lm_id]
            max_x_lm = landmarker_coordinates[self.max_x_lm_id]
            min_y_lm = landmarker_coordinates[self.min_y_lm_id]
            max_y_lm = landmarker_coordinates[self.max_y_lm_id]
            cx = int(round((min_x_lm[0] + max_x_lm[0])/2.0))
            cy = int(round((min_y_lm[1] + max_y_lm[1])/2.0))
            
            # Generate the rectangle
            masked_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            x1 = max(0, min_x_lm[0] - self.bar_padding)
            x2 = min(w-1, max_x_lm[0] + self.bar_padding)
            y1 = max(0, cy - self.bar_padding)
            y2 = min(h-1, cy + self.bar_padding)
            cv.rectangle(masked_frame, (x1, y1), (x2, y2), (255,255,255), -1)
            
            # Generate rotation matrix and rotate the rectangle
            rot_mat = cv.getRotationMatrix2D((cx,cy), (rot_angle), 1.0)
            rot_mask = cv.warpAffine(masked_frame, rot_mat, (w,h), flags=cv.INTER_NEAREST)
            rot_mask = rot_mask.astype(bool)

            output_frame = frame.copy().astype(np.uint8)
            output_frame[rot_mask] = self.bar_color
            
            return output_frame

def layer_occlusion_bar(timing_configuration:TimingConfiguration | None = None, landmark_path:list[tuple[int,...]] = LANDMARK_BOTH_EYE_REGIONS, bar_colour:tuple[int,int,int] = (0,0,0), bar_padding:int = 50) -> LayerOcclusionBar:
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = BarOcclusionParameters(
            bar_colour=bar_colour, 
            bar_padding=bar_padding,
            landmark_path=landmark_path
        )
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerOcclusionBar.__name__}: {e}")

    return LayerOcclusionBar(time_config, params)