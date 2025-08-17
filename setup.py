import os
import subprocess
import sys
from pathlib import Path
from pathlib import PurePosixPath
from scripts.dir import list_dir, find_extra_item, copy_folder_if_exists
sys.path.append("/content/Colab_CV")
from google.colab import output

script_dir = Path(__file__).resolve().parent

yolov7_dir = script_dir / "yolov7"
gr_requirements_path = script_dir / "gr_req.txt"

print(f"Installing dependencies from: {gr_requirements_path}")
subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(gr_requirements_path)], check=True)

import gradio as gr


# ---- define your pipeline ----
def run_pipeline(user_url, api_key, version, width, epochs, batch):
    try:
        from helper import install_deps, pull_dataset, train_model
        install_deps()
        project_name = pull_dataset(user_url, api_key, int(version))
        train_model(int(width), int(epochs), int(batch), project_name)
        return f"✅ Training finished for project {project_name}!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ---- Gradio frontend ----
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 YOLOv7 Training Frontend (Colab Auto-Launch)")

    with gr.Row():
        user_url = gr.Textbox(label="Roboflow Project URL")
        api_key = gr.Textbox(label="API Key", type="password")
        version = gr.Number(label="Dataset Version", value=1)

    with gr.Row():
        width = gr.Number(label="Image Width", value=416)
        epochs = gr.Number(label="Epochs", value=10)
        batch = gr.Number(label="Batch Size", value=16)

    run_btn = gr.Button("▶️ Start Training")
    output_box = gr.Textbox(label="Logs / Status", lines=10)

    run_btn.click(run_pipeline, [user_url, api_key, version, width, epochs, batch], output_box)


# ---- launch and auto-open in new window ----
port = 7860
demo.launch(server_name="0.0.0.0", server_port=port, share=False, inbrowser=False)

# force Colab to open it
output.eval_js(f"window.open('http://127.0.0.1:{port}')")