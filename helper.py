import os
import subprocess
import sys
from pathlib import Path
from pathlib import PurePosixPath
from scripts.dir import list_dir, find_extra_item, copy_folder_if_exists, copy_file_if_exists

script_dir = Path(__file__).resolve().parent

yolov7_dir = script_dir / "yolov7"


# Get absolute path to the directory where this script is located

def install_deps(): 

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
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements_path)], check=True)

    print("Checking for GPU...")
    try:
        import torch
        if not torch.cuda.is_available():
            print(" No GPU detected. We highly recommend that you go to runtime and switch the runtime type to either a T4 or L4 before continuing.")
            input("Warning: Make sure you change your runtime to GPU. Press Enter to continue...")
    except ImportError:
        print("Installing torch...")
        subprocess.run(["pip", "install", "torch"], check=True)

    copy_folder_if_exists (yolov7_dir, "models" , script_dir / "scripts")
    copy_folder_if_exists (yolov7_dir, "utils" , script_dir / "scripts")

def pull_dataset(user_url, api_key, version): 
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



    try:
        workspaco, projecto = parse_roboflow_url(user_url)
        print("Workspace:", workspaco)
        print("Project:", projecto)
    except ValueError as e:
        print("Error:", e)

    from roboflow import Roboflow

    os.chdir(yolov7_dir)


    items = list_dir(yolov7_dir)
    rf = Roboflow(api_key)
    proj = rf.workspace(workspaco).project(projecto)
    dataset = proj.version(version).download("yolov5")

    new_items = list_dir(yolov7_dir)
    
    project_name = find_extra_item(items, new_items)
    return project_name
def patch_yolov7_weights_only(yolov7_dir, make_backups=True, verbose=True):
    """
    Patch all torch.load(...) calls in YOLOv7 to include weights_only=False,
    and fix any accidental 'torch.device(\"cpu\", weights_only=False)' inserts.

    Args:
        yolov7_dir (str | pathlib.Path): Path to the YOLOv7 repository folder.
        make_backups (bool): If True, writes a .py.bak backup next to each modified file.
        verbose (bool): If True, prints per-file and summary info.

    Returns:
        (files_patched:int, calls_patched:int)
    """
    from pathlib import Path

    yolov7_dir = Path(yolov7_dir)
    if not yolov7_dir.is_dir():
        raise FileNotFoundError(f"yolov7_dir not found: {yolov7_dir}")

    def _fix_text(src):
        # Clean up bad prior patch: torch.device('cpu', weights_only=False) -> torch.device('cpu')
        cleaned = src.replace("torch.device('cpu', weights_only=False)", "torch.device('cpu')")
        cleaned = cleaned.replace('torch.device("cpu", weights_only=False)', 'torch.device("cpu")')

        needle = "torch.load("
        L = len(needle)
        i = 0
        n = len(cleaned)
        out_parts = []
        changed_calls = 0

        while True:
            j = cleaned.find(needle, i)
            if j == -1:
                out_parts.append(cleaned[i:])
                break

            # Copy up to "torch.load("
            out_parts.append(cleaned[i:j])
            out_parts.append(needle)

            # Find matching ')', handling nested parens and strings
            k = j + L
            depth = 1
            in_str = False
            quote = ""
            escape = False

            while k < n and depth > 0:
                ch = cleaned[k]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == quote:
                        in_str = False
                else:
                    if ch in ("'", '"'):
                        in_str = True
                        quote = ch
                    elif ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                k += 1

            args = cleaned[j + L : k - 1]  # inside the parentheses

            # Add weights_only=False if missing
            if "weights_only" not in args:
                if args.strip():
                    args = args.rstrip() + ", weights_only=False"
                else:
                    args = "weights_only=False"
                changed_calls += 1

            out_parts.append(args)
            out_parts.append(")")
            i = k

        new_text = "".join(out_parts)
        return new_text, changed_calls

    files_patched = 0
    calls_patched = 0

    for path in yolov7_dir.rglob("*.py"):
        original = path.read_text(encoding="utf-8")
        fixed, calls = _fix_text(original)
        if fixed != original:
            if make_backups:
                backup = path.with_suffix(path.suffix + ".bak")
                backup.write_text(original, encoding="utf-8")
            path.write_text(fixed, encoding="utf-8")
            files_patched += 1
            calls_patched += calls
            if verbose:
                rel = path.relative_to(yolov7_dir)
                print(f"Patched {rel}  (+{calls} torch.load call(s))")

    if verbose:
        if files_patched == 0:
            print("No changes made (already patched or patterns not found).")
        else:
            print(f"Done. Files patched: {files_patched}, torch.load calls updated: {calls_patched}")
            if make_backups:
                print("Backups saved as *.py.bak next to each modified file.")

    return files_patched, calls_patched

def train_model(width, epochs, batch, project_name):
   
    data_path = f"/content/Colab_CV/yolov7/{project_name}/data.yaml"

    
    
    train_command = [
        "python", "train.py",
        "--img", str(width), "320",
        "--batch", str(batch),
        "--epochs", str(epochs),
        "--data", data_path,
        "--cfg", "cfg/training/yolov7-tiny.yaml",
        "--weights", "yolov7-tiny.pt",
        "--name", "yolov7-tiny-amb82",
        "--workers", "8"
    ]

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"  # disables logging to Weights & Biases


    subprocess.run(train_command, env=env, cwd="/content/Colab_CV/yolov7", check=True, text=True)

def download_model():
    model_file = yolov7_dir / "runs" / "train" / "yolov7-tiny-amb82" / "weights" / "best.pt"
    return str(model_file) if model_file.exists() else None

def amb82mini_reparam():
    model_folder = yolov7_dir / "runs" / "train" / "yolov7-tiny-amb82" / "weights"
    copy_file_if_exists(model_folder, "best.pt" , script_dir / "scripts")
    os.chdir(script_dir / "scripts")
    ckpt = torch.load("best.pt", map_location="cpu")
    model = ckpt['model']
    nc = model.nc  
    def update_yaml_nc(yaml_path, nc):
        lines = []
        with open(yaml_path, "r") as f:
            for line in f:
                if line.strip().startswith("nc:"):
                    lines.append(f"nc: {nc}\n")  
                    lines.append(line)
        with open(yaml_path, "w") as f:
            f.writelines(lines)
    update_yaml_nc("yolov7-tiny-deploy.yaml", nc)

    subprocess.run(["python3", "reparam_yolov7-tiny.py", "--weights", "best.pt", "--custom_yaml" , "yolov7-tiny-deploy.yaml", "--output", "best_reparam.pt", "--nc", str(nc) ])
