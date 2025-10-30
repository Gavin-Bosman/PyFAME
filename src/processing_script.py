### TODO

# Look for some new tears (datasets or individual samples), and additional overlay assets (beard, moustache, mask, piercing)
# Update pytest test suite

# Colour, Mask, Occlusion, spatial, overlay, layers now work as expected (+ Analysis methods)
# Don't forget to go through stylise and temporal layers, and conversion methods

# consider pre-processing pass finding frames where blinks occur, similar to accurate_colour_scale with Optical flow methods
# Consider looking into passing both the unaltered and current (altered) frame to each layer in LayerPipeline
# If I havent already, weigh pupil size by timestamp in pupil overlay.

import pyfame as pf

file_paths = pf.make_paths()
of_result = pf.analyse_facial_colour_means(file_paths.iloc[[1]], colour_space="hsv")
pf.analysis_to_disk(of_result, "Facial Colour Means")