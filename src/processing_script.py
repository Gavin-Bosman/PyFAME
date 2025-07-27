import pyfame as pf

root_data_folder = ".\\ravdess_data\\Video_Song_Actors_01-24\\Video_Song_Actor_01\\Actor_01"

# Create input and output subdirectories ['raw','processed',...]
paths = pf.make_paths(exclude_directories=["processed", "logs"])

optical_flow = pf.layer_stylise_optical_flow_sparse(time_onset=1.5, time_offset=3.0)

pf.apply_layers(paths, [optical_flow])

### TODO test dense optical flow layer
### Test all layers performing as expected