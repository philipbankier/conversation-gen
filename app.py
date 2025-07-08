"""
Kyutai TTS (1.6 B) → MultiTalk (14 B, 480 p) Space
Runs on a single L4 (24 GB GPU).
"""
# --------------------------------------------------------------------------- #
# 0 · Bootstrap imports + version guards
# --------------------------------------------------------------------------- #
import os, sys, gc, subprocess, pathlib, uuid, torch, soundfile as sf, gradio as gr
import importlib.metadata as md

GRADIO_VER   = "5.35.0"
CLIENT_MIN   = "1.10.0"

def _ensure(pkg: str, spec: str):
    try:
        if not eval(f'md.version("{pkg}") {spec!s}'):
            raise md.PackageNotFoundError
    except md.PackageNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{pkg}{spec}"])
        os.execv(sys.executable, [sys.executable] + sys.argv)  # hard-restart

_ensure("gradio", f"=={GRADIO_VER}")
_ensure("gradio_client", f">={CLIENT_MIN}")

# --------------------------------------------------------------------------- #
# 1 · Runtime clones / heavy deps
# --------------------------------------------------------------------------- #
MULTITALK_DIR = pathlib.Path("multitalk_src")
if not MULTITALK_DIR.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/MeiGen-AI/MultiTalk.git",
                           str(MULTITALK_DIR)])
sys.path.append(str(MULTITALK_DIR.resolve()))
from multitalk_wrapper import generate_video           # noqa: E402

# --------------------------------------------------------------------------- #
# 2 · Kyutai & Moshi
# --------------------------------------------------------------------------- #
from huggingface_hub import list_repo_files
from moshi.models import loaders
from moshi.models.tts import TTSModel

TTS_REPO   = "kyutai/tts-1.6b-en_fr"
VOICE_REPO = "kyutai/tts-voices"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

def synthesize(text: str, voice_id: str):
    if not hasattr(synthesize, "_codec"):
        print("🔄 Loading Kyutai Mimi & TTS weights …")
        mimi_ckpt          = loaders.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        synthesize._codec  = loaders.get_mimi(mimi_ckpt, device=DEVICE)
        ckpt_info          = loaders.CheckpointInfo.from_hf_repo(TTS_REPO)
        synthesize._tts    = TTSModel.from_checkpoint_info(
                                  ckpt_info, n_q=32, temp=0.6, device=torch.device(DEVICE))
    return synthesize._tts(text, voice=voice_id,
                           codec=synthesize._codec,
                           sample_rate=loaders.SAMPLE_RATE).astype("float32")

def voice_catalog():
    if not hasattr(voice_catalog, "_cache"):
        files = list_repo_files(VOICE_REPO); suf = ".safetensors"
        voice_catalog._cache = sorted(f[:-len(suf)] for f in files if f.endswith(suf))
    return voice_catalog._cache

# --------------------------------------------------------------------------- #
# 3 · Gemini 2.5-pro via google-genai
# --------------------------------------------------------------------------- #
import genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEM_MODEL = "gemini-2.5-pro-preview-05-06"

def make_prompt(img_name: str, l1: str, l2: str) -> str:
    model  = genai.GenerativeModel(GEM_MODEL)
    prompt = (f"Craft one vivid, single-sentence scene description for an AI video generator.\n"
              f"The still image depicts: {img_name}.\n"
              f"The dialogue is:\nSpeaker 1: {l1}\nSpeaker 2: {l2}")
    return model.generate_content(prompt, timeout=20).text.strip()

# --------------------------------------------------------------------------- #
# 4 · Main inference pipeline
# --------------------------------------------------------------------------- #
def run(line1, line2, image, v1, v2):
    wav1, wav2 = (synthesize(line1, v1), synthesize(line2, v2))
    sf.write("spk1.wav", wav1, 24_000); sf.write("spk2.wav", wav2, 24_000)
    torch.cuda.empty_cache(); gc.collect()

    scene_prompt = make_prompt(getattr(image, "name", "an image"), line1, line2)
    return generate_video(image, ["spk1.wav", "spk2.wav"], scene_prompt)

# --------------------------------------------------------------------------- #
# 5 · Gradio 5.x UI
# --------------------------------------------------------------------------- #
with gr.Blocks(title="Kyutai TTS → MultiTalk (480 p)") as demo:
    gr.Markdown("### Kyutai TTS → MultiTalk\nUpload an image and two lines of text; get a 480 p talking-head clip.")
    l1, l2 = gr.Text(label="Speaker 1"), gr.Text(label="Speaker 2")
    img    = gr.Image(type="filepath", label="Reference image")
    vopts  = voice_catalog()
    v1     = gr.Dropdown(vopts, label="Voice 1", value=vopts[0] if vopts else None)
    v2     = gr.Dropdown(vopts, label="Voice 2", value=vopts[0] if vopts else None)
    out    = gr.Video(label="Output video")
    gr.Button("Generate").click(run, [l1, l2, img, v1, v2], out, concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(share=False, max_threads=8)
