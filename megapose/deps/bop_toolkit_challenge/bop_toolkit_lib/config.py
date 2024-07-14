# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague
import os
from pathlib import Path



"""Configuration of the BOP Toolkit."""

######## Basic ########

# Folder with the BOP datasets.
datasets_path = str(os.environ['BOP_DATASETS_PATH'])
results_path = str(os.environ['BOP_RESULTS_PATH'])
eval_path = str(os.environ['BOP_EVAL_PATH'])


######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r'/path/to/output/folder'

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/path/to/bop_renderer/build'

# Executable of the MeshLab server.
meshlab_server_path = r'/path/to/meshlabserver.exe'
