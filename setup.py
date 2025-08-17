import os
import subprocess
import sys
from pathlib import Path
from pathlib import PurePosixPath
from scripts.dir import list_dir, find_extra_item, copy_folder_if_exists
sys.path.append("/content/Colab_CV")
from helper import install_deps

install_deps()