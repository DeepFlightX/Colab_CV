

import os, sys, subprocess
def ensure_python311():
    import platform
    major, minor, _ = platform.python_version_tuple()
    if major == "3" and minor != "11":
        print("Switching runtime to Python 3.11 ...")

        # Install Python 3.11 and tools
        subprocess.run(["apt-get", "update", "-y"], check=True)
        subprocess.run(["apt-get", "install", "-y", "python3.11", "python3.11-distutils"], check=True)

        # Make python3 point to python3.11
        subprocess.run([
            "update-alternatives", "--install", "/usr/bin/python3", "python3", "/usr/bin/python3.11", "1"
        ], check=True)
        subprocess.run(["update-alternatives", "--set", "python3", "/usr/bin/python3.11"], check=True)

        # Re-exec the script with Python 3.11
        os.execv("/usr/bin/python3.11", ["python3.11"] + sys.argv)

ensure_python311()
# Install pip for Python 3.11
subprocess.run(["curl", "-sS", "https://bootstrap.pypa.io/get-pip.py", "-o", "get-pip.py"], check=True)
subprocess.run(["/usr/bin/python3.11", "get-pip.py"], check=True)


import os, sys, subprocess
from pathlib import Path

sys.path.append("/content/Colab_CV")

# --- robust script_dir (works in .py and in a notebook cell) ---
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()
import os, sys, subprocess




yolov7_dir = script_dir / "yolov7"
gr_requirements_path = script_dir / "gr_req.txt"

# --- install all frontend deps ---
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-cache-dir",
     "-r", str(gr_requirements_path), "--constraint", str(gr_requirements_path)],
    check=True
)


# --- ensure Pillow version matches runtime ---
def ensure_pillow(target="10.4.0"):
    try:
        from importlib.metadata import version
        import PIL, PIL._imaging as _img  # noqa
        if version("Pillow") != target or getattr(_img, "__version__", None) != target:
            raise ImportError("Pillow mismatch")
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               f"Pillow=={target}", "--force-reinstall", "--no-cache-dir"])
        if os.environ.get("REEXEC_ONCE") != "1":
            os.environ["REEXEC_ONCE"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)

ensure_pillow("10.4.0")

import gradio as gr

# ---- pipeline ----
def run_pipeline(user_url, api_key, version, width, epochs, batch):
    try:
        from helper import install_deps, pull_dataset, train_model, download_model, patch_yolov7_weights_only, amb82mini_reparam
        install_deps()
        patch_yolov7_weights_only("/content/Colab_CV/yolov7")
        project_name = pull_dataset(user_url, api_key, int(version))
        train_model(int(width), int(epochs), int(batch), project_name)
        model_path = download_model()
        amb82mini_reparam()
        return f" Training finished for project {project_name}!", model_path
        
    except Exception as e:
        return f" Error: {e}", None
        

# ---- Gradio UI ----
def prepare_download(path):
    """Prepare a DownloadButton with the file path (no Colab files.download here)."""
    if path and os.path.exists(path):
        # set button's 'value' to the filepath and show it
        return gr.update(value=path, visible=True), "Ready"
    return gr.update(visible=False, value=None), "No model found yet"

with gr.Blocks() as demo:
    gr.Markdown("# YOLOv7 Training Frontend (Colab)")

    with gr.Row():
        user_url = gr.Textbox(label="Roboflow Project URL")
        api_key  = gr.Textbox(label="API Key", type="password")
        version  = gr.Number(label="Dataset Version", value=1, precision=0)

    with gr.Row():
        width  = gr.Number(label="Image Width", value=416, precision=0)
        epochs = gr.Number(label="Epochs", value=10, precision=0)
        batch  = gr.Number(label="Batch Size", value=16, precision=0)

    run_btn     = gr.Button("Start Training")
    output_box  = gr.Textbox(label="Logs / Status", lines=10)

    # Keep the model path in state (not visible)
    model_path_state = gr.State(value=None)

    # New: button that “activates” the download control
    make_dl_btn  = gr.Button("Make Download Button")
    # New: Download button that actually serves best.pt when clicked
    download_btn = gr.DownloadButton(label="Download best.pt", visible=False)

    # Training returns (status_text, model_path_str)
    run_btn.click(
        run_pipeline,
        inputs=[user_url, api_key, version, width, epochs, batch],
        outputs=[output_box, model_path_state],
    )

    # User clicks to prepare the download button after training
    # We also echo a small status next to it in a textbox
    dl_status = gr.Textbox(label="Download Status", interactive=False)

    make_dl_btn.click(
        prepare_download,
        inputs=[model_path_state],
        outputs=[download_btn, dl_status],
    )

# Important: disable the API docs route that triggers the schema bug
demo.launch(share=True, show_api=False)

