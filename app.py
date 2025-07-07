"""
Kyutai TTS (1.6 B)  ➜  MultiTalk (14 B)  • 480 p • single L4 (24 GB)
"""
# --------------------------------------------------------------------------- #
# 0.  bootstrap: install gradio + client *without* pulling their deps
# --------------------------------------------------------------------------- #
import subprocess, sys, os, importlib.metadata as md
def _pip_no_deps(pkg_spec: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--no-cache-dir", "--no-deps", pkg_spec])

try:
    if md.version("gradio") < "4.44.0":
        print("🔄  Installing gradio 4.49.0 (no-deps)…")
        _pip_no_deps("gradio==4.49.0")
except md.PackageNotFoundError:
    _pip_no_deps("gradio==4.49.0")

try:
    if md.version("gradio_client") < "1.10.0":
        print("🔄  Installing gradio_client 1.10.4 (no-deps)…")
        _pip_no_deps("gradio_client==1.10.4")
except md.PackageNotFoundError:
    _pip_no_deps("gradio_client==1.10.4")

# --------------------------------------------------------------------------- #
# 1.  std imports (now gradio is safely importable)
# --------------------------------------------------------------------------- #
import gc, pathlib, uuid, json, numpy as np
import torch, soundfile as sf, gradio as gr, genai
from huggingface_hub import list_repo_files
from moshi.models import loaders
from moshi.models.tts import TTSModel

# --------------------------------------------------------------------------- #
# 2.  clone MultiTalk once
# --------------------------------------------------------------------------- #
MULTITALK_DIR = pathlib.Path("multitalk_src")
if not MULTITALK_DIR.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/MeiGen-AI/MultiTalk.git",
                           str(MULTITALK_DIR)])
import sys; sys.path.append(str(MULTITALK_DIR))
from multitalk_wrapper import generate_video             # noqa: E402

# --------------------------------------------------------------------------- #
# 3.  constants
# --------------------------------------------------------------------------- #
TTS_REPO   = "kyutai/tts-1.6b-en_fr"
VOICE_REPO = "kyutai/tts-voices"
GEM_MODEL  = "gemini-2.5-pro-preview-05-06"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# --------------------------------------------------------------------------- #
# 4.  Kyutai single-load synthesiser
# --------------------------------------------------------------------------- #
def synthesize(text: str, voice_id: str) -> np.ndarray:
    if not hasattr(synthesize, "_state"):
        print("🔄 Loading Mimi + TTS weights…")
        mimi_w   = loaders.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi     = loaders.get_mimi(mimi_w, device=DEVICE)

        ckpt     = loaders.CheckpointInfo.from_hf_repo(TTS_REPO)
        tts      = TTSModel.from_checkpoint_info(ckpt, n_q=32, temp=0.6,
                                                 device=torch.device(DEVICE))
        synthesize._state = (mimi, tts)

    mimi, tts = synthesize._state
    wav = tts(text, voice=voice_id, codec=mimi, sample_rate=loaders.SAMPLE_RATE)
    return wav.astype("float32")

# --------------------------------------------------------------------------- #
# 5.  helpers
# --------------------------------------------------------------------------- #
def voice_catalog():
    if not hasattr(voice_catalog, "_cache"):
        files = list_repo_files(VOICE_REPO)
        voice_catalog._cache = sorted(f[:-11] for f in files if f.endswith(".safetensors"))
    return voice_catalog._cache

def gemini_prompt(img_name: str, l1: str, l2: str) -> str:
    model  = genai.GenerativeModel(GEM_MODEL)
    prompt = (f"Craft one vivid single-sentence scene description for an AI video "
              f"generator. The still image depicts: {img_name}. Dialogue:\n"
              f"Speaker 1: {l1}\nSpeaker 2: {l2}")
    return model.generate_content(prompt, stream=False).text.strip()

# --------------------------------------------------------------------------- #
# 6.  pipeline
# --------------------------------------------------------------------------- #
def pipeline(text_a, text_b, image_path, voice_a, voice_b):
    wav_a = synthesize(text_a, voice_a); sf.write("spk1.wav", wav_a, 24_000)
    wav_b = synthesize(text_b, voice_b); sf.write("spk2.wav", wav_b, 24_000)

    torch.cuda.empty_cache(); gc.collect()

    prompt = gemini_prompt(os.path.basename(image_path), text_a, text_b)
    return generate_video(image_path, ["spk1.wav", "spk2.wav"], prompt)

# --------------------------------------------------------------------------- #
# 7.  UI
# --------------------------------------------------------------------------- #
with gr.Blocks(title="Kyutai TTS ➜ MultiTalk (480 p)") as demo:
    gr.Markdown("## One image + two lines → talking-head video")
    t1, t2      = gr.Text(label="Speaker 1"), gr.Text(label="Speaker 2")
    img         = gr.Image(label="Reference image", type="filepath")
    vlist       = voice_catalog()
    va, vb      = gr.Dropdown(vlist, value=vlist[0], label="Voice 1"), \
                  gr.Dropdown(vlist, value=vlist[1], label="Voice 2")
    out         = gr.Video(label="Generated MP4")
    btn         = gr.Button("Generate")
    btn.click(pipeline, [t1, t2, img, va, vb], out, concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(share=False, max_threads=10)
