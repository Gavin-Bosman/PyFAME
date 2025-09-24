### TODO
# Come back to image_overlay function after investigating new mediapipe.tasks facemesh
# and playing with varying confidence param values

# Investigate how masking, occlusion, and overlay interact when applied to the same file

import pyfame as pf

paths = pf.make_paths()
time_cf = pf.TimingConfiguration(time_onset=0.5, time_offset=3.0)

recolour = pf.layer_colour_recolour(time_cf, pf.CHEEKS_NOSE_PATH, magnitude=12.0)
blur = pf.layer_occlusion_blur(time_cf, blur_method="average")
mask = pf.layer_mask(time_cf, region_of_interest=pf.FACE_OVAL_PATH)

pf.apply_layers(paths.iloc[[0]], [mask])