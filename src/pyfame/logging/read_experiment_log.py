import json
import os
import jsonschema
from jsonschema import ValidationError
from importlib.resources import files
from pyfame.layer.layer import Layer, TimingConfiguration
from pyfame.layer.manipulations import *
import pyfame.layer.timing_curves as t_curves
import pyfame.utilities.constants as const
import pyfame.mesh.mesh_landmarks as landmark

def read_experiment_log(log_file_path:str) -> list[Layer]:
    if not os.path.exists(log_file_path):
        raise ValueError("Invalid log file path provided, log file Cannot be read.")
    if not os.path.isfile(log_file_path):
        raise IsADirectoryError("Log file path exists, but does not point to a file.")
    if not log_file_path.lower().endswith(".json"):
        raise ValueError(f"Expected a .json log file, got: {log_file_path}")
    
    const_dict = vars(const)
    lm_dict = vars(landmark)
    timing_dict = vars(t_curves)

    # Layer mappings
    layer_name_map = {
        "LayerColourBrightness":layer_colour_brightness,
        "LayerColourRecolour":layer_colour_recolour,
        "LayerColourSaturation":layer_colour_saturation,
        "LayerMask":layer_mask,
        "LayerOcclusionBar":layer_occlusion_bar,
        "LayerOcclusionPath":layer_occlusion_path,
        "LayerOcclusionBlur":layer_occlusion_blur,
        "LayerOcclusionNoise":layer_occlusion_noise,
        "LayerSpatialGridShuffle":layer_spatial_grid_shuffle,
        "LayerSpatialLandmarkRelocate":layer_spatial_landmark_relocate,
        "LayerStylisePointLight":layer_stylise_point_light,
        "LayerStyliseOverlay":layer_stylise_overlay,
        "LayerStyliseOpticalFlowDense":layer_stylise_optical_flow_dense,
        "LayerStyliseOpticalFlowSparse":layer_stylise_optical_flow_sparse
    }
    
    # Read in the json
    with open(log_file_path, "r") as fp:
        experiment_log = json.load(fp)
    
    # Validate input log against schema
    try:
        schema_path = files("pyfame.schema").joinpath("manipulation_log.v1.schema.json")
        schema = json.load(open(schema_path))
        jsonschema.validate(instance=experiment_log, schema=schema)
    except ValidationError as e:
        raise ValueError(f"Experiment log failed to validate: {e.message}.")
    
    layers_return = []
    layers = experiment_log.get("layers")
    layer_names = list(layers.keys())

    for name in layer_names:
        fn = layer_name_map.get(name)
        params = layers.get(name)

        onset = params.pop("time_onset")
        offset = params.pop("time_offset")
        timing_func = params.pop("timing_function")
        rise = params.pop("rise_duration")
        fall = params.pop("fall_duration")
        mtc = params.pop("min_tracking_confidence")
        mdc = params.pop("min_detection_confidence")

        timeconfig = TimingConfiguration(
            time_onset=onset,
            time_offset=offset,
            timing_function=timing_dict.get(timing_func),
            rise_duration=rise,
            fall_duration=fall,
            min_tracking_confidence=mtc,
            min_detection_confidence=mdc
        )

        # Resolve remaining params
        for k,v in params.items():
            if isinstance(v, str):
                if v in const_dict:
                    params.update({k:getattr(const, v)})
                elif v in lm_dict:
                    params.update({k:getattr(landmark, v)})

        layer_instance = fn(timeconfig, **params)
        layers_return.append(layer_instance)
    
    return layers_return