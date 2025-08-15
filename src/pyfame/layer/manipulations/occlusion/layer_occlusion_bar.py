from pydantic import BaseModel, field_validator, ValidationError, ValidationInfo
from typing import Union, Tuple, List
from pyfame.utilities.general_utilities import compute_rot_angle
from pyfame.utilities.constants import *
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.mesh import *
import cv2 as cv
import numpy as np

class BarOcclusionParameters(BaseModel):
    bar_colour:Tuple[int,int,int]
    region_of_interest:Union[List[List[Tuple[int,...]]], List[Tuple[int,...]]]

    @field_validator("bar_colour")
    @classmethod
    def check_in_range(cls, value, info:ValidationInfo):
        field_name = info.field_name
        for elem in value:
            if not (0 <= elem <= 255):
                raise ValueError(f"{field_name} values must lie between 0 and 255.")
    
    @field_validator("region_of_interest", mode='before')
    @classmethod
    def check_compatible_path(cls, value, info:ValidationInfo):
        valid_paths = [LEFT_EYE_PATH, RIGHT_EYE_PATH, BOTH_EYES_PATH, NOSE_PATH, LIPS_PATH, MOUTH_PATH]
        field_name = info.field_name

        if isinstance(value[0], List):
            for sublist in value:
                if sublist not in valid_paths:
                    raise ValueError(f"Incompatible path provided in {field_name}. Please provide one of: LEFT_EYE_PATH, RIGHT_EYE_PATH, BOTH_EYES_PATH, NOSE_PATH, LIPS_PATH, MOUTH_PATH.")
                
        elif value not in valid_paths:
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
        self.region_of_interest = self.occlude_params.region_of_interest
        self.min_x_lm_id = -1
        self.max_x_lm_id = -1
        self.static_image_mode = False
        self.min_detection_confidence = self.time_config.min_detection_confidence
        self.min_tracking_confidence = self.time_config.min_tracking_confidence

        # Dump the pydantic models to get full param list
        self._layer_parameters = self.time_config.model_dump()
        self._layer_parameters.update(self.occlude_params.model_dump())
            
    def supports_weight(self):
        return False
    
    def get_layer_parameters(self):
        return dict(self._layer_parameters)
    
    def apply_layer(self, frame:cv.typing.MatLike, dt:float, static_image_mode:bool = False):

        # Update faceMesh when switching between image and video processing
        if static_image_mode != self.static_image_mode:
            self.static_image_mode = static_image_mode
            super().set_face_mesh(self.min_tracking_confidence, self.min_detection_confidence, self.static_image_mode)
        
        # Bar occlusion does not support weight, so weight will always be 0.0 or 1.0
        weight = super().compute_weight(dt, self.supports_weight())

        if weight == 0.0:
            return frame
        else:
            # Get the faceMesh coordinate set
            face_mesh = super().get_face_mesh()
            lm_coords = get_mesh_coordinates(cv.cvtColor(frame, cv.COLOR_BGR2RGB), face_mesh)
            masked_frame = np.zeros_like(frame, dtype=np.uint8)
            refactored_lms = []
            
            # Replace placeholder concave path with its convex sub-paths
            if isinstance(self.region_of_interest[0], list):
                for lm in self.region_of_interest:
                    if lm == BOTH_EYES_PATH:
                        refactored_lms.append(LEFT_EYE_PATH)
                        refactored_lms.append(RIGHT_EYE_PATH)
                    else:
                        refactored_lms.append(lm)
            elif self.region_of_interest == BOTH_EYES_PATH:
                refactored_lms.append(LEFT_EYE_PATH)
                refactored_lms.append(RIGHT_EYE_PATH)

            for lm in refactored_lms:

                min_x = 1000
                max_x = 0

                # find the two points closest to the beginning and end x-positions of the landmark region
                unique_landmarks = np.unique(lm)
                for lm_id in unique_landmarks:
                    cur_lm = lm_coords[lm_id]
                    if cur_lm.get('x') < min_x:
                        min_x = cur_lm.get('x')
                        self.min_x_lm_id = lm_id
                    if cur_lm.get('x') > max_x:
                        max_x = cur_lm.get('x')
                        self.max_x_lm_id = lm_id

            # Calculate the slope of the connecting line & angle to the horizontal
            p1 = lm_coords[self.min_x_lm_id]
            p2 = lm_coords[self.max_x_lm_id]
            slope = (p2.get('y') - p1.get('y'))/(p2.get('x') - p1.get('x'))
            rot_angle = compute_rot_angle(slope_1=slope)
            
            # Compute the center bisecting line of the landmark
            cx = round((p2.get('y') + p1.get('y'))/2)
            cy = round((p2.get('x') + p1.get('x'))/2)
            
            # Generate the rectangle
            rectangle = cv.rectangle(masked_frame, (p1.get('x')-50, cx - 50), (p2.get('x') + 50, cx + 50), (255,255,255), -1)
            masked_frame_t = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
            
            # Generate rotation matrix and rotate the rectangle
            rot_mat = cv.getRotationMatrix2D((cy,cx), (rot_angle), 1)
            rot_img = cv.warpAffine(rectangle, rot_mat, (masked_frame_t.shape[1], masked_frame_t.shape[0]))
            
            masked_frame = cv.bitwise_or(masked_frame, np.where(rot_img == 255, 255, masked_frame_t))
            
            output_frame = np.where(masked_frame == 255, self.bar_color, frame)
            return output_frame.astype(np.uint8)

def layer_occlusion_bar(timing_configuration:TimingConfiguration | None = None, region_of_interest:list[list[tuple[int,...]]] | list[tuple[int,...]] = FACE_OVAL_PATH, bar_colour:tuple[int,int,int] = (0,0,0)) -> LayerOcclusionBar:
    # Populate with defaults if None
    time_config = timing_configuration or TimingConfiguration()

    # Validate input parameters
    try:
        params = BarOcclusionParameters(bar_colour, region_of_interest)
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for {LayerOcclusionBar.__name__}: {e}")

    return LayerOcclusionBar(time_config, params)