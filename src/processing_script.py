import pyfame as pf

# Create input and output subdirectories ['raw','processed',...]
paths = pf.make_paths()
timing = pf.TimingConfiguration()

recolour = pf.layer_colour_recolour(timing, pf.CHIN_PATH, magnitude=15.0)

pf.apply_layers(paths.iloc[[0]], recolour)