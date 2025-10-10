### TODO
# wrap up facemesh conversion 
# Rework object overlay
    # Look for some new tears (datasets or individual samples)
    # Explore eye/iris related overlays
# Update pytest test suite
# Revisit enforce_layer_ordering(), experimentally determine an order of operations
# Compare current nose landmarks to new mediapipe.tasks landmark sets

# Colour, Mask, Occlusion layers now work as expected

#                                           !!!!EXPLORE FALL DURATION BUGS WITH CHATGPT!!!!


import pyfame as pf

file_paths = pf.make_paths()

config = pf.TimingConfiguration(timing_function=pf.timing_linear, rise_duration=0.75, fall_duration=0.75)

# mask = pf.layer_mask(config, pf.LANDMARK_FACE_OVAL)
# brighten = pf.layer_colour_brightness(config, pf.LANDMARK_FACE_OVAL)
# recolour = pf.layer_colour_recolour(config, pf.LANDMARK_CHEEKS_AND_NOSE, magnitude=15.0)
# occlude = pf.layer_occlusion_landmark(config, [pf.LANDMARK_LEFT_EYE, pf.LANDMARK_RIGHT_EYE, pf.LANDMARK_NOSE, pf.LANDMARK_LIPS])
# bar_occlude = pf.layer_occlusion_bar(config)
# grid_shuffle = pf.layer_spatial_grid_shuffle(config, random_seed=1234, shuffle_method=pf.HIGH_LEVEL_GRID_SHUFFLE)
# landmark_shuffle = pf.layer_spatial_landmark_relocate(config, random_seed=1234)
# overlay = pf.layer_stylise_overlay(config, overlay_name_or_path="tear_short_1")
sclera = pf.layer_colour_redden_sclera(config)

pf.apply_layers(file_paths, sclera)