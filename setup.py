import os
import subprocess
import sys
from pathlib import Path
from pathlib import PurePosixPath


# Get absolute path to the directory where this script is located
script_dir = Path(__file__).resolve().parent

yolov7_dir = script_dir / "yolov7"

if not yolov7_dir.exists():
    print(f"Cloning yolov7 into: {yolov7_dir}")
    subprocess.run([
        "git", "clone", "https://github.com/WongKinYiu/yolov7.git", str(yolov7_dir)
    ], check=True)
else:
    print("yolov7 repo already exists, skipping clone.")

# Build absolute path to requirements.txt
requirements_path = script_dir / "requirements.txt"

print(f"Installing dependencies from: {requirements_path}")
subprocess.run(["pip", "install", "-r", str(requirements_path)], check=True)

print("Checking for GPU...")
try:
    import torch
    if not torch.cuda.is_available():
        print(" No GPU detected. We highly recommend that you go to runtime and switch the runtime type to either a T4 or L4 before continuing.")
        input("Warning: Make sure you change your runtime to GPU. Press Enter to continue...")
except ImportError:
    print("Installing torch...")
    subprocess.run(["pip", "install", "torch"], check=True)

def parse_roboflow_url(url: str):
    # Remove trailing slash
    cleaned = url.rstrip('/')

    # Remove everything before and including 'roboflow.com/'
    if 'roboflow.com/' in cleaned:
        post = cleaned.split('roboflow.com/', 1)[1]
    else:
        raise ValueError("URL does not contain 'roboflow.com/'")

    # Break up the path into parts
    path = PurePosixPath(post)
    parts = path.parts

    if len(parts) >= 2:
        workspaco, projecto = parts[0], parts[1]
    else:
        raise ValueError("URL does not contain both workspace and project names.")

    return workspaco, projecto


user_url = input("Paste your Roboflow project URL: ").strip()
api_key = input ("Input your api key from: ").strip()
version = int (input ("Version, input the version number as an integer")) 

try:
    workspaco, projecto = parse_roboflow_url(user_url)
    print("Workspace:", workspaco)
    print("Project:", projecto)
except ValueError as e:
    print("Error:", e)

from roboflow import Roboflow

os.chdir(yolov7_dir)

rf = Roboflow(api_key)
proj = rf.workspace(workspaco).project(projecto)
dataset = proj.version(version).download("yolov5")

# width = int (input("Set image width for "))
train_command = [
    "python", "train.py",
    "--img", "576", "320",
    "--batch", "128",
    "--epochs", "50",
    "--data", "/content/Colab_CV/yolov7/Drone-1/data.yaml",
    "--cfg", "cfg/training/yolov7-tiny.yaml",
    "--weights", "yolov7-tiny.pt",
    "--name", "yolov7-tiny-amb82",
    "--workers", "8"
]

env = os.environ.copy()
env["WANDB_MODE"] = "disabled"  # disables logging to Weights & Biases


subprocess.run(train_command, env=env, cwd="/content/Colab_CV/yolov7", check=True, text=True)