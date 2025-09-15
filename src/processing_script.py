import pyfame as pf

# Create input and output subdirectories ['raw','processed',...]
paths = pf.make_paths()
timing = pf.TimingConfiguration()

flow = pf.layer_stylise_optical_flow_dense(timing, precise_colour_scale=True)

pf.apply_layers(paths.iloc[[1]], flow)