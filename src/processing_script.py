import pyfame as pf

# Create input and output subdirectories ['raw','processed',...]
paths = pf.make_paths()
timing = pf.TimingConfiguration()

mask = pf.layer_mask(timing)

pf.apply_layers(paths, mask)

# Progress bars for:
# write log file
# Layer-level execution