"""
Kyutai TTS (1.6 B) → MultiTalk (14 B, 480 p)  –  fits on one L4.
"""
# ---------------------------------------------------------------------------#
# 0. Imports + one-time Gradio patch
# ---------------------------------------------------------------------------#
import os, sys, gc, subprocess, pathlib, torch, numpy as np, soundfile as sf, gradio as gr
import genai                                              # new Google GenAI SDK
from huggingface_hub import list_repo_files
from moshi.models import loaders
from moshi.models.tts import TTSModel
import importlib.metadata as md

GRADIO_TARGET = "4.44.1"
if md.version("gradio") != GRADIO_TARGET:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           f"gradio=={GRADIO_TARGET}", "--no-cache-dir",
                           "--force-reinstall"])
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ---------------------------------------------------------------------------#
# 1.  Clone MultiTalk repo (binary ops unavailable on pip)
# ---------------------------------------------------------------------------#
MULTITALK_DIR = pathlib.Path("multitalk_src")
if not MULTITALK_DIR.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/MeiGen-AI/MultiTalk.git",
                           str(MULTITALK_DIR)])
sys.path.append(str(MULTITALK_DIR.resolve()))
from multitalk_wrapper import generate_video                              # noqa

# ---------------------------------------------------------------------------#
# 2. Constants
# ---------------------------------------------------------------------------#
TTS_REPO   = "kyutai/tts-1.6b-en_fr"
VOICE_REPO = "kyutai/tts-voices"
GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------------------------------------------------------------#
# 3. Speech synthesis
# ---------------------------------------------------------------------------#
def synthesize(text: str, voice_id: str) -> np.ndarray:
    if not hasattr(synthesize, "_codec"):
        print("🔄  Loading Kyutai Mimi + TTS …")
        mimi_ckpt           = loaders.hf_hub_download(loaders.DEFAULT_REPO,
                                                     loaders.MIMI_NAME)
        synthesize._codec   = loaders.get_mimi(mimi_ckpt, device=DEVICE)
        ckpt_info           = loaders.CheckpointInfo.from_hf_repo(TTS_REPO)
        synthesize._tts     = TTSModel.from_checkpoint_info(
                                  ckpt_info, n_q=32, temp=0.6,
                                  device=torch.device(DEVICE))

    # --- Kyutai generation pipeline ---
    entries    = synthesize._tts.prepare_script([text])
    voice_path = synthesize._tts.get_voice_path(voice_id)
    cond_attrs = synthesize._tts.make_condition_attributes([voice_path], cfg_coef=2.0)

    with synthesize._tts.mimi.streaming(1), torch.no_grad():
        out = synthesize._tts.generate([entries], [cond_attrs])

    frames = [
        synthesize._tts.mimi.decode(f[:, 1:, :]).cpu().numpy()[0, 0]
        for f in out.frames[synthesize._tts.delay_steps :]
    ]
    return np.concatenate(frames).astype("float32")  # 24 kHz PCM

# ---------------------------------------------------------------------------#
# 4. Helpers
# ---------------------------------------------------------------------------#
_voice_cache = None
def get_voice_catalog():
    global _voice_cache
    if _voice_cache is None:
        suffix = ".safetensors"
        _voice_cache = sorted(
            f[:-len(suffix)] for f in list_repo_files(VOICE_REPO) if f.endswith(suffix)
        )
    return _voice_cache

def make_scene_prompt(img_name, a, b):
    gm = genai.GenerativeModel(GEMINI_MODEL)
    prompt = (f"Craft one vivid, single-sentence scene description for an AI video generator.\n"
              f"The still image depicts: {img_name}.\n"
              f"The dialogue is:\nSpeaker 1: {a}\nSpeaker 2: {b}")
    return gm.generate_content(prompt, timeout=20).text.strip()

# ---------------------------------------------------------------------------#
# 5. Pipeline
# ---------------------------------------------------------------------------#
def infer(text_a, text_b, image, voice_a, voice_b):
    wav_a = synthesize(text_a, voice_a)
    wav_b = synthesize(text_b, voice_b)
    sf.write("spk1.wav", wav_a, 24_000)
    sf.write("spk2.wav", wav_b, 24_000)

    torch.cuda.empty_cache(); gc.collect()

    prompt = make_scene_prompt(getattr(image, "name", "an image"), text_a, text_b)
    return generate_video(image, ["spk1.wav", "spk2.wav"], prompt)

# ---------------------------------------------------------------------------#
# 6. UI
# ---------------------------------------------------------------------------#
with gr.Blocks(title="Kyutai TTS → MultiTalk (480 p)") as demo:
    gr.Markdown("# Kyutai TTS → MultiTalk\nTwo lines + one image → talking-head clip (480 p).")
    with gr.Row():
        text_a = gr.Text(label="Speaker 1 text")
        text_b = gr.Text(label="Speaker 2 text")
    image_in = gr.Image(label="Reference image", type="filepath")
    voices   = get_voice_catalog()
    with gr.Row():
        voice_a = gr.Dropdown(label="Voice 1", choices=voices, value=voices[0])
        voice_b = gr.Dropdown(label="Voice 2", choices=voices, value=voices[0])
    out_vid = gr.Video(label="Generated clip")
    gr.Button("Generate").click(infer,
                                inputs=[text_a, text_b, image_in, voice_a, voice_b],
                                outputs=out_vid,
                                concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(max_threads=10)
