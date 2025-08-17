import os
import subprocess
import sys
from pathlib import Path
from pathlib import PurePosixPath
from scripts.dir import list_dir, find_extra_item, copy_folder_if_exists
sys.path.append("/content/Colab_CV")
from helper import install_deps, pull_dataset, train_model

script_dir = Path(__file__).resolve().parent

yolov7_dir = script_dir / "yolov7"

install_deps()

user_url = input("Paste your Roboflow project URL: ").strip() 
api_key = input ("Input your api key from: ").strip() 
version = int (input ("Version, input the version number as an integer"))

pull_dataset(user_url, api_key, version)

width = int (input("Set image width for CV input")) 
epochs = int (input ("Choose number of epochs")) 
batch = int (input ("Choose number images per batch"))

train_model(width, epochs, batch)