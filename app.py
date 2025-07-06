"""
Kyutai TTS (1.6 B) → MultiTalk (14 B, 480 p) Space
Runs on a single L4 (24 GB) GPU.
"""

# ---------------------------------------------------------------------------#
# 0.  Standard + runtime-bootstrap imports
# ---------------------------------------------------------------------------#
import os, sys, gc, subprocess, pathlib, uuid, json, torch, soundfile as sf, gradio as gr
import google.generativeai as genai
from huggingface_hub import list_repo_files
from moshi.models import loaders                           # lower-level API

# ---------------------------------------------------------------------------#
# 1.  Clone MultiTalk at runtime (pip-install would fail)
# ---------------------------------------------------------------------------#
MULTITALK_DIR = pathlib.Path("multitalk_src")
if not MULTITALK_DIR.exists():
    print("🔄  Cloning MultiTalk repo…")
    subprocess.check_call(
        ["git", "clone", "--depth", "1",
         "https://github.com/MeiGen-AI/MultiTalk.git",
         str(MULTITALK_DIR)]
    )
    # optional: try flash-attn now that torch is importable (fails gracefully)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "flash-attn==2.6.1", "--no-build-isolation"])
    except subprocess.CalledProcessError:
        print("⚠️  flash-attn build failed — falling back to default attention")

sys.path.append(str(MULTITALK_DIR.resolve()))
from multitalk_wrapper import generate_video               # noqa: E402

# ---------------------------------------------------------------------------#
# 2.  Constants
# ---------------------------------------------------------------------------#
TTS_REPO      = "kyutai/tts-1.6b-en_fr"
VOICE_REPO    = "kyutai/tts-voices"
GEMINI_MODEL  = "gemini-2.5-pro-preview-05-06"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------------------------------------------------------------#
# 3.  Kyutai synthesis helper (replacement for missing moshi.TTS)
# ---------------------------------------------------------------------------#
def synthesize(text: str, voice_id: str):
    """
    Generate a 24 kHz float32 waveform for `text` in the style of `voice_id`.
    Caches Mimi codec and TTS weights after first load.
    """
    if not hasattr(synthesize, "_codec"):
        print("🔄  Loading Kyutai Mimi + TTS weights …")
        # Mimi codec
        mimi_ckpt = loaders.hf_hub_download(loaders.DEFAULT_REPO,
                                            loaders.MIMI_NAME)
        synthesize._codec = loaders.get_mimi(mimi_ckpt, device=DEVICE)
        # TTS model
        tts_ckpt = loaders.hf_hub_download(TTS_REPO, "pytorch_model.bin")
        synthesize._tts = loaders.get_tts(tts_ckpt, device=DEVICE)

    wav = synthesize._tts(
        text,
        voice=voice_id,
        codec=synthesize._codec,
        sample_rate=loaders.SAMPLE_RATE
    )
    return wav.astype("float32")          # numpy array @ 24 kHz

# ---------------------------------------------------------------------------#
# 4.  Voice preset catalogue
# ---------------------------------------------------------------------------#
_voice_cache = None
def get_voice_catalog():
    global _voice_cache
    if _voice_cache is None:
        files   = list_repo_files(VOICE_REPO)
        suffix  = ".safetensors"
        _voice_cache = sorted(
            f[:-len(suffix)] for f in files if f.endswith(suffix)
        )
    return _voice_cache

# ---------------------------------------------------------------------------#
# 5.  Gemini prompt helper
# ---------------------------------------------------------------------------#
def make_scene_prompt(img_name: str, line_a: str, line_b: str) -> str:
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
# 6.  Core inference pipeline
# ---------------------------------------------------------------------------#
def infer(text_a, text_b, image, voice_a, voice_b):
    # 6.1  Speech synthesis (GPU ~6 GB)
    wav_a = synthesize(text_a, voice_a)
    wav_b = synthesize(text_b, voice_b)
    sf.write("spk1.wav", wav_a, 24_000)
    sf.write("spk2.wav", wav_b, 24_000)

    # 6.2  Free VRAM before spinning up MultiTalk
    torch.cuda.empty_cache(); gc.collect()

    # 6.3  Gemini prompt
    prompt = make_scene_prompt(getattr(image, "name", "an image"), text_a, text_b)

    # 6.4  MultiTalk → MP4
    return generate_video(image, ["spk1.wav", "spk2.wav"], prompt)

# ---------------------------------------------------------------------------#
# 7.  Gradio UI
# ---------------------------------------------------------------------------#
with gr.Blocks(title="Kyutai TTS → MultiTalk (480 p)") as demo:
    gr.Markdown(
        "# Kyutai TTS → MultiTalk\n"
        "Two lines of dialogue + a reference image → talking-head clip (480 p, ≈15 s)."
    )

    with gr.Row():
        text_a = gr.Text(label="Speaker 1 text")
        text_b = gr.Text(label="Speaker 2 text")
    image_in = gr.Image(label="Reference image", type="filepath")

    catalog       = get_voice_catalog()
    default_voice = catalog[0] if catalog else None
    with gr.Row():
        voice_a = gr.Dropdown(label="Voice 1", choices=catalog, value=default_voice)
        voice_b = gr.Dropdown(label="Voice 2", choices=catalog, value=default_voice)

    out_vid  = gr.Video(label="Generated clip")
    gen_btn  = gr.Button("Generate")
    gen_btn.click(infer, [text_a, text_b, image_in, voice_a, voice_b], out_vid)

if __name__ == "__main__":
    demo.queue(concurrency_count=1, max_size=5).launch()
