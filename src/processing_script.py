import pyfame as pf

# Create input and output subdirectories ['raw','processed',...]
paths = pf.make_paths(exclude_directories=["processed", "logs", "conversion"])

# Optionally, define a custom timing configuration
timing_config = pf.TimingConfiguration(time_onset=0.5, time_offset=3.0, rise_duration=0.25, fall_duration=0.75)

# Define Layers
recolour = pf.layer_colour_recolour(timing_config, pf.CHEEKS_NOSE_PATH, "blue", 12.0)
# Notice we didnt pass timing config here; it will be populated with defaults internally
mask = pf.layer_mask(background_colour=(255,255,255))

# Best practice to apply weighted operations prior to non-weighted operations,
# however layer_pipeline cleanly handles this internally so layer order in 
# apply_layers call doesnt actually matter
pf.apply_layers(paths, [recolour, mask])