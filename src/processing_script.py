import pyfame as pf

# Create input and output subdirectories ['raw','processed',...]
paths = pf.make_paths(exclude_directories=["processed", "logs", "conversion"])

# Optionally, define a custom timing configuration
timing_config = pf.TimingConfiguration(time_onset=0.0, time_offset=3.0)

# Define Layers
overlay = pf.layer_stylise_overlay(timing_config, overlay_name_or_path="teardrop")

# Best practice to apply weighted operations prior to non-weighted operations,
# however layer_pipeline cleanly handles this internally so layer order in 
# apply_layers call doesnt actually matter
pf.apply_layers(paths, [overlay])