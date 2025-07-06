"""
Kyutai TTS → MultiTalk (480 p) Space
Runs on a single L4 (24 GB) GPU.
"""

# ---------------------------------------------------------------------------#
# 0.  Standard imports
# ---------------------------------------------------------------------------#
import os, sys, gc, subprocess, pathlib, uuid, torch, soundfile as sf, gradio as gr
import google.generativeai as genai
from huggingface_hub import list_repo_files
from moshi import TTS
from multitalk_wrapper import generate_video

# ---------------------------------------------------------------------------#
# 1.  Clone MultiTalk (and optionally flash-attn) at *runtime*
#     – avoids pip-install errors during Space build
# ---------------------------------------------------------------------------#
MULTITALK_DIR = pathlib.Path("multitalk_src")
if not MULTITALK_DIR.exists():
    print("🔄  Cloning MultiTalk repo…")
    subprocess.check_call(
        ["git", "clone", "--depth", "1",
         "https://github.com/MeiGen-AI/MultiTalk.git",
         str(MULTITALK_DIR)]
    )

    # ❶ optional: build flash-attn now that torch is importable
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install",
             "flash-attn==2.6.1", "--no-build-isolation"]
        )
        print("✅ flash-attn installed")
    except subprocess.CalledProcessError:
        print("⚠️  flash-attn build failed — falling back to default attention")

# make `import multitalk` resolve
sys.path.append(str(MULTITALK_DIR.resolve()))

# ---------------------------------------------------------------------------#
# 2.  Constants
# ---------------------------------------------------------------------------#
TTS_REPO     = "kyutai/tts-1.6b-en_fr"
VOICE_REPO   = "kyutai/tts-voices"
GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"

tts = None           # lazy-initialised Kyutai synthesiser
voice_ids = None     # preset catalogue cache

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------------------------------------------------------------#
# 3.  Helper functions
# ---------------------------------------------------------------------------#
def get_voice_catalog():
    """Fetch list of voice IDs from kyutai/tts-voices repo (cached)."""
    global voice_ids
    if voice_ids is None:
        files   = list_repo_files(VOICE_REPO)
        suffix  = ".safetensors"
        voice_ids = sorted(
            f[:-len(suffix)] for f in files if f.endswith(suffix)
        )
    return voice_ids

def make_scene_prompt(img_name: str, line_a: str, line_b: str) -> str:
    """Ask Gemini for a one-sentence scene description."""
    gm     = genai.GenerativeModel(GEMINI_MODEL)
    prompt = (
        "Craft one vivid, single-sentence scene description for an AI video generator.\n"
        f"The still image depicts: {img_name}.\n"
        "The dialogue is:\n"
        f"Speaker 1: {line_a}\n"
        f"Speaker 2: {line_b}"
    )
    rsp = gm.generate_content(prompt, timeout=20)
    return rsp.text.strip()

# ---------------------------------------------------------------------------#
# 4.  Core inference pipeline
# ---------------------------------------------------------------------------#
def infer(text_a, text_b, image, voice_a, voice_b):
    global tts

    # --- 4.1 Kyutai TTS (lazy init once) -------------------------------
    if tts is None:
        tts = TTS(repo=TTS_REPO, fp16=True)

    wav_a = tts(text_a, voice=voice_a)
    wav_b = tts(text_b, voice=voice_b)
    sf.write("spk1.wav", wav_a, 24_000)
    sf.write("spk2.wav", wav_b, 24_000)

    # --- 4.2 Free VRAM before launching MultiTalk ----------------------
    del tts            # truly release CUDA buffers
    torch.cuda.empty_cache()
    gc.collect()

    # --- 4.3 Gemini scene prompt --------------------------------------
    prompt = make_scene_prompt(getattr(image, "name", "an image"), text_a, text_b)

    # --- 4.4 MultiTalk video generation -------------------------------
    return generate_video(image, ["spk1.wav", "spk2.wav"], prompt)

# ---------------------------------------------------------------------------#
# 5.  Gradio UI
# ---------------------------------------------------------------------------#
with gr.Blocks(title="Kyutai TTS → MultiTalk (480 p)") as demo:
    gr.Markdown(
        "# Kyutai TTS → MultiTalk\n"
        "Upload an image, type two lines of dialogue, choose voices → "
        "get a 15-second 480 p talking-head clip."
    )

    with gr.Row():
        text_a = gr.Text(label="Speaker 1 text")
        text_b = gr.Text(label="Speaker 2 text")

    image_in = gr.Image(label="Reference image", type="filepath")

    voices       = get_voice_catalog()
    default_voice = voices[0] if voices else None
    with gr.Row():
        voice_a = gr.Dropdown(label="Voice 1", choices=voices, value=default_voice)
        voice_b = gr.Dropdown(label="Voice 2", choices=voices, value=default_voice)

    out_video = gr.Video(label="Generated Clip")

    generate_btn = gr.Button("Generate")
    generate_btn.click(
        infer,
        inputs=[text_a, text_b, image_in, voice_a, voice_b],
        outputs=out_video
    )

# ---------------------------------------------------------------------------#
# 6.  Launch
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    demo.queue(concurrency_count=1, max_size=5).launch()
