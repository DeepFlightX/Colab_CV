import os, sys, subprocess
from pathlib import Path
sys.path.append("/content/Colab_CV")

# --- robust script_dir (works in .py and in a notebook cell) ---
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()

yolov7_dir = script_dir / "yolov7"
gr_requirements_path = script_dir / "gr_req.txt"

# --- install all frontend deps, and pin transitive ones via a constraint ---
# (prevents Pillow from being silently upgraded by sub-dependencies)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-cache-dir",
     "-r", str(gr_requirements_path), "--constraint", str(gr_requirements_path)],
    check=True
)

# --- ensure Pillow exactly matches runtime (prevents _imaging mismatch); re-exec once if needed ---
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
from google.colab import output as colab_output  # IMPORTANT: only works from the notebook kernel

# ---- pipeline (uses your helper.py) ----
def run_pipeline(user_url, api_key, version, width, epochs, batch):
    try:
        from helper import install_deps, pull_dataset, train_model
        install_deps()
        project_name = pull_dataset(user_url, api_key, int(version))
        train_model(int(width), int(epochs), int(batch), project_name)
        return f" Training finished for project {project_name}!"
    except Exception as e:
        return f" Error: {e}"

# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("#  YOLOv7 Training Frontend (Colab Auto-Launch)")
    with gr.Row():
        user_url = gr.Textbox(label="Roboflow Project URL")
        api_key  = gr.Textbox(label="API Key", type="password")
        version  = gr.Number(label="Dataset Version", value=1, precision=0)
    with gr.Row():
        width  = gr.Number(label="Image Width", value=416, precision=0)
        epochs = gr.Number(label="Epochs", value=10, precision=0)
        batch  = gr.Number(label="Batch Size", value=16, precision=0)
    run_btn    = gr.Button(" Start Training")
    output_box = gr.Textbox(label="Logs / Status", lines=10)
    run_btn.click(run_pipeline, [user_url, api_key, version, width, epochs, batch], output_box)

# ---- launch + auto-open ----
PORT = 7860
demo.launch(server_name="0.0.0.0", server_port=PORT, share=False, prevent_thread_lock=True)
colab_output.serve_kernel_port_as_window(PORT)   # or: colab_output.serve_kernel_port_as_iframe(PORT, height=650)
