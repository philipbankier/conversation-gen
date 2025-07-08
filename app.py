"""
Kyutai TTS (1.6 B) → MultiTalk (14 B, 480 p)
Runs on a single L4 (24 GB GPU).
"""

# --------------------------------------------------------------------------- #
# 0 · Bootstrap imports + version guards
# --------------------------------------------------------------------------- #
import os, sys, gc, subprocess, pathlib, torch, soundfile as sf, gradio as gr
import importlib.metadata as md
from packaging import version as pv               # ← safer version handling

GRADIO_VER = "5.35.0"      # exact
CLIENT_MIN = "1.10.0"      # at least

def _ensure(pkg: str, spec: str):
    """
    spec is either '==X.Y.Z'  or  '>=A.B.C'.
    If the currently installed version does not satisfy the spec,
    install/upgrade the package and hard-restart the interpreter.
    """
    try:
        cur = md.version(pkg)
    except md.PackageNotFoundError:
        cur = None

    need = spec[2:]
    satisfied = False
    if spec.startswith("=="):
        satisfied = (cur == need)
    elif spec.startswith(">=") and cur is not None:
        satisfied = pv.parse(cur) >= pv.parse(need)

    if not satisfied:
        print(f"🔄  Installing {pkg}{spec} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               f"{pkg}{spec}", "--upgrade", "--no-cache-dir"])
        os.execv(sys.executable, [sys.executable, *sys.argv])  # hard restart

_ensure("gradio",        f"=={GRADIO_VER}")
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

def synthesize(txt: str, voice: str):
    if not hasattr(synthesize, "_codec"):
        print("🔄  Loading Kyutai Mimi + TTS weights …")
        mimi_ckpt         = loaders.hf_hub_download(loaders.DEFAULT_REPO,
                                                   loaders.MIMI_NAME)
        synthesize._codec = loaders.get_mimi(mimi_ckpt, device=DEVICE)
        ckpt_info         = loaders.CheckpointInfo.from_hf_repo(TTS_REPO)
        synthesize._tts   = TTSModel.from_checkpoint_info(
                                ckpt_info, n_q=32, temp=0.6,
                                device=torch.device(DEVICE))
    wav = synthesize._tts(txt, voice=voice,
                          codec=synthesize._codec,
                          sample_rate=loaders.SAMPLE_RATE)
    return wav.astype("float32")            # 24 kHz NumPy array

def voice_catalog():
    if not hasattr(voice_catalog, "_cache"):
        suf = ".safetensors"
        voice_catalog._cache = sorted(
            f[:-len(suf)] for f in list_repo_files(VOICE_REPO) if f.endswith(suf))
    return voice_catalog._cache

# --------------------------------------------------------------------------- #
# 3 · Gemini 2.5-pro via google-genai
# --------------------------------------------------------------------------- #
import genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEM_MODEL = "gemini-2.5-pro-preview-05-06"

def scene_prompt(img_name: str, l1: str, l2: str) -> str:
    m = genai.GenerativeModel(GEM_MODEL)
    p = (f"Craft one vivid, single-sentence scene description for an AI video generator.\n"
         f"The still image depicts: {img_name}.\n"
         f"Dialogue:\nSpeaker 1: {l1}\nSpeaker 2: {l2}")
    return m.generate_content(p, timeout=20).text.strip()

# --------------------------------------------------------------------------- #
# 4 · Main inference pipeline
# --------------------------------------------------------------------------- #
def run(line1, line2, image, v1, v2):
    wav1, wav2 = synthesize(line1, v1), synthesize(line2, v2)
    sf.write("spk1.wav", wav1, 24_000); sf.write("spk2.wav", wav2, 24_000)
    torch.cuda.empty_cache(); gc.collect()
    prompt = scene_prompt(getattr(image, "name", "an image"), line1, line2)
    return generate_video(image, ["spk1.wav", "spk2.wav"], prompt)

# --------------------------------------------------------------------------- #
# 5 · Gradio 5.x UI
# --------------------------------------------------------------------------- #
with gr.Blocks(title="Kyutai TTS → MultiTalk (480 p)") as demo:
    gr.Markdown("### Kyutai TTS → MultiTalk\nUpload an image + two lines of text; receive a 480 p talking-head clip.")
    l1, l2 = gr.Text(label="Speaker 1"), gr.Text(label="Speaker 2")
    img    = gr.Image(type="filepath", label="Reference image")
    vopts  = voice_catalog()
    v1     = gr.Dropdown(vopts, label="Voice 1", value=vopts[0] if vopts else None)
    v2     = gr.Dropdown(vopts, label="Voice 2", value=vopts[0] if vopts else None)
    out    = gr.Video(label="Output video")
    gr.Button("Generate").click(run, [l1, l2, img, v1, v2], out,
                                 concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(share=False, max_threads=8)
