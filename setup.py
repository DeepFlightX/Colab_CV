import os
import sys
import subprocess
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import time

# ---------- Runtime safety toggles ----------
# Avoid jax_plugins trying to import Triton
os.environ["JAX_DISABLE_PLUGIN_DISCOVERY"] = "1"
# Disable W&B logs
os.environ["WANDB_MODE"] = "disabled"

SCRIPT_DIR = Path(__file__).resolve().parent
Y7_DIR = SCRIPT_DIR / "yolov7"
WEIGHTS_PATH = Y7_DIR / "yolov7-tiny.pt"

def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, text=True, **kwargs)

def pip_install(args):
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *args])

def ensure_torch_cu121():
    # Install CUDA 12.1 wheels for L4 (GPU). If you’re CPU-only, remove the --index-url.
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return
    except Exception:
        pass
    print("\nInstalling PyTorch (CUDA 12.1) ...")
    pip_install([
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "torch==2.1.2",
        "torchvision==0.16.2",
    ])
    import torch
    print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())

def ensure_yolov7_repo():
    if not Y7_DIR.exists():
        print(f"Cloning yolov7 into: {Y7_DIR}")
        run(["git", "clone", "https://github.com/WongKinYiu/yolov7.git", str(Y7_DIR)])
    else:
        print("yolov7 repo already exists, skipping clone.")
    # Make sure submodules (not strictly required)
    try:
        run(["git", "-C", str(Y7_DIR), "submodule", "update", "--init", "--recursive"])
    except Exception:
        pass

def download_file(url: str, dest: Path, chunk: int = 1 << 20):
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url} -> {dest}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for b in r.iter_content(chunk_size=chunk):
                if b:
                    f.write(b)

def ensure_yolov7_tiny_weights():
    if WEIGHTS_PATH.exists():
        print("Found yolov7-tiny weights.")
        return
    # Official release URL
    url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
    download_file(url, WEIGHTS_PATH)

def parse_roboflow_url(url: str):
    cleaned = url.strip().rstrip('/')
    key = 'roboflow.com/'
    if key not in cleaned:
        raise ValueError("URL must contain 'roboflow.com/'. Example: https://universe.roboflow.com/<workspace>/<project>")
    post = cleaned.split(key, 1)[1]
    parts = PurePosixPath(post).parts
    if len(parts) < 2:
        raise ValueError("URL must include both <workspace> and <project>.")
    return parts[0], parts[1]

def find_data_yaml(search_root: Path) -> Path | None:
    # Roboflow SDK typically returns .location dir with data.yaml inside
    for p in search_root.rglob("data.yaml"):
        # prefer files that have 'data.yaml' directly under dataset folder
        return p
    return None

def main():
    print("Ensuring CUDA-capable PyTorch...")
    ensure_torch_cu121()

    print("Ensuring yolov7 repo...")
    ensure_yolov7_repo()
    ensure_yolov7_tiny_weights()

    # --- Gather inputs ---
    print("\nPaste your Roboflow project URL (e.g., https://universe.roboflow.com/<workspace>/<project>):")
    user_url = input("> ").strip()
    print("Input your Roboflow API key:")
    api_key = input("> ").strip()
    print("Enter dataset version number (integer):")
    version_str = input("> ").strip()
    try:
        version = int(version_str)
    except ValueError:
        raise SystemExit("Version must be an integer.")

    # --- Parse RF URL ---
    workspaco, projecto = parse_roboflow_url(user_url)
    print("Workspace:", workspaco)
    print("Project:", projecto)

    # --- Download dataset via Roboflow ---
    from roboflow import Roboflow

    print("\nDownloading Roboflow dataset...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspaco).project(projecto)
    dataset = proj.version(version).download("yolov5")  # yolov7 uses the same format

    # Determine data.yaml path
    data_root = Path(getattr(dataset, "location", SCRIPT_DIR))
    data_yaml = find_data_yaml(data_root)
    if data_yaml is None:
        # Fall back to searching under repo dir
        data_yaml = find_data_yaml(Y7_DIR)
    if data_yaml is None:
        raise SystemExit("Could not find data.yaml after download.")
    print("Using data.yaml:", data_yaml)

    # --- Build training command ---
    # Use a single --img value (square). If you need rectangular, modify the YAML and loaders.
    IMG = "576"
    workers = str(min(8, (os.cpu_count() or 2)))

    train_cmd = [
        sys.executable, "train.py",
        "--img", IMG,
        "--batch", "128",
        "--epochs", "50",
        "--data", str(data_yaml),
        "--cfg", "cfg/training/yolov7-tiny.yaml",
        "--weights", str(WEIGHTS_PATH),
        "--name", "yolov7-tiny-amb82",
        "--workers", workers,
    ]

    # --- Inform about GPU ---
    import torch
    if not torch.cuda.is_available():
        print("\n[Warning] No CUDA GPU detected by PyTorch.")
        print("  In Colab, set Runtime → Change runtime type → GPU (prefer T4/L4).")
    else:
        print("CUDA GPU detected. Device count:", torch.cuda.device_count())

    # --- Launch training ---
    print("\nLaunching YOLOv7 training...")
    env = os.environ.copy()
    # Ensure we run inside repo so relative paths work
    run(train_cmd, cwd=str(Y7_DIR), env=env)

    print("\nTraining finished.")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("\n[train_yolov7_roboflow.py] A subprocess failed.")
        print("Command:", e.cmd)
        print("Return code:", e.returncode)
        sys.exit(e.returncode)
    except Exception as exc:
        print("\n[train_yolov7_roboflow.py] Error:", repr(exc))
        sys.exit(1)
