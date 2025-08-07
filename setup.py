import os
import subprocess
import sys

print("Cloning repos...")
os.system("git clone https://github.com/WongKinYiu/yolov7.git")

print("Installing dependencies")
subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)



print("Checking for GPU...")
try:
    import torch
    if not torch.cuda.is_available():
        print(" No GPU detected, we highly recommend that you go to runtime, and switch the runtime type to either a T4 or L4 before continuing")
except ImportError:
    print("Installing torch...")
    subprocess.run(["pip", "install", "torch"])




