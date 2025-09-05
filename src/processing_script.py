import pyfame as pf

# Create input and output subdirectories ['raw','processed',...]
paths = pf.make_paths()
timing = pf.TimingConfiguration()

flow = pf.layer_stylise_optical_flow_dense(timing)

pf.apply_layers(paths, flow)