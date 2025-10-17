### TODO

# Look for some new tears (datasets or individual samples), and additional overlay assets (beard, moustache, mask, piercing)
# Update pytest test suite

# Colour, Mask, Occlusion, spatial, layers now work as expected
# Don't forget to go through analysis methods, stylise and temporal layers, and conversion methods
# consider pre-processing pass finding frames where blinks occur, similar to accurate_colour_scale with Optical flow methods

import pyfame as pf

file_paths = pf.make_paths()

config = pf.TimingConfiguration(timing_function=pf.timing_constant)
recolour = pf.layer_colour_recolour(config, [pf.LANDMARK_BOTH_EYES], magnitude=12.0)
overlay = pf.layer_overlay(config, overlay_type=pf.OVERLAY_PUPIL_DILATION)

pf.apply_layers(file_paths, [recolour, overlay])