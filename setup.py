import os
import subprocess
import sys
from pathlib import Path

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