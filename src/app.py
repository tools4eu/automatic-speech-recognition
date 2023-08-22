import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from transcribe.transcribe import transcriber, languages
import gradio as gr
import torch
import torchaudio
import torch.cuda as cuda
import platform
from transformers import __version__ as transformers_version


device = "cuda" if torch.cuda.is_available() else "cpu"
num_gpus = cuda.device_count() if torch.cuda.is_available() else 0
cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
cudnn_version = torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A"
os_info = platform.system() + " " + platform.release() + " " + platform.machine()

# Get the available VRAM for each GPU (if available)
vram_info = []
if torch.cuda.is_available():
    for i in range(cuda.device_count()):
        gpu_properties = cuda.get_device_properties(i)
        vram_info.append(f"GPU {i}: {gpu_properties.total_memory / 1024**3:.2f} GB")

pytorch_version = torch.__version__
torchaudio_version = torchaudio.__version__ if 'torchaudio' in dir() else "N/A"

device_info = f"""Running on: {device}
    Number of GPUs available: {num_gpus}
    CUDA version: {cuda_version}
    CuDNN version: {cudnn_version}
    PyTorch version: {pytorch_version}
    Torchaudio version: {torchaudio_version}
    Transformers version: {transformers_version}
    Operating system: {os_info}
    Available VRAM: 
    \t {', '.join(vram_info) if vram_info else 'N/A'}
"""

# def inference(input, diarize, num_speakers:int, strict, lan, trans, progress=gr.Progress()):
def inference(input, lan, trans):
    tr = transcriber(input, lan, translate=trans)
    return {textbox: gr.update(value= tr["text"])}

with gr.Blocks(title="Automatic speech recognition") as demo:
    with gr.Row():
        gr.Markdown(
            """
                # Automatic speech recognition

                [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
            """
        )
    with gr.Tab("Upload sound"):
        with gr.Row():
            with gr.Column():
                upl_input = gr.Audio(type='filepath')
                upl_language = gr.Dropdown(
                    label='Language', 
                    choices = ['INFERRED']+sorted(list(languages.keys())), 
                    value='INFERRED', 
                    info="""
                        Setting the language to "INFERRED" will auto-detect the language based on the first 30 seconds.
                        If the language is known upfront, always set it manually.
                    """)

        with gr.Row():
            upl_translate = gr.Checkbox(label='Translate to English')
            
        with gr.Row():
            upl_btn = gr.Button("Transcribe it")
        
        with gr.Row(variant='panel'):
            with gr.Column():
                textbox = gr.Textbox(label='Transciption',visible=True)
    with gr.Tab("Device info"):
        gr.Textbox(device_info, label="Hardware info & installed packages", lines=len(device_info.split("\n")))

    transcribe_event = upl_btn.click(fn =inference, inputs=[upl_input, upl_language, upl_translate], outputs=[textbox])

demo.queue(concurrency_count=1)
demo.launch()