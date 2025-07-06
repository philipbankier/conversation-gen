import os, gc, json, subprocess, uuid, tempfile, torch, soundfile as sf, gradio as gr
import google.generativeai as genai
from huggingface_hub import list_repo_files
from moshi import TTS
from multitalk_wrapper import generate_video

# ---- Constants -------------------------------------------------------------
TTS_REPO   = "kyutai/tts-1.6b-en_fr"
VOICE_REPO = "kyutai/tts-voices"
GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"

tts       = None          # lazy-init Kyutai synthesiser
voice_ids = None          # cache of preset voices
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---- Helpers ---------------------------------------------------------------
def get_voice_catalog():
    """Return sorted list of <folder>/<file-stem> voice IDs."""
    global voice_ids
    if voice_ids is not None:
        return voice_ids
    files      = list_repo_files(VOICE_REPO)
    suffix     = ".safetensors"
    voice_ids  = sorted(f[:-len(suffix)] for f in files if f.endswith(suffix))
    return voice_ids

def make_scene_prompt(img_name: str, line_a: str, line_b: str) -> str:
    gm     = genai.GenerativeModel(GEMINI_MODEL)
    prompt = f"""
    Craft one vivid, single-sentence scene description for an AI video generator.
    The still image depicts: {img_name}.
    The dialogue is:
    Speaker 1: {line_a}
    Speaker 2: {line_b}
    """
    rsp = gm.generate_content(prompt, timeout=20)
    return rsp.text.strip()

# ---- Core pipeline ---------------------------------------------------------
def infer(text_a, text_b, image, voice_a, voice_b):
    global tts

    # 1. Init Kyutai TTS once
    if tts is None:
        tts = TTS(repo=TTS_REPO, fp16=True)

    # 2. Synthesize WAVs
    wav_a = tts(text_a, voice=voice_a)
    wav_b = tts(text_b, voice=voice_b)
    sf.write("spk1.wav", wav_a, 24_000)
    sf.write("spk2.wav", wav_b, 24_000)

    # 3. Free GPU memory before MultiTalk
    del tts
    torch.cuda.empty_cache()
    gc.collect()

    # 4. Gemini scene prompt
    prompt = make_scene_prompt(getattr(image, "name", "an image"), text_a, text_b)

    # 5. Generate video
    return generate_video(image, ["spk1.wav", "spk2.wav"], prompt)

# ---- Gradio UI -------------------------------------------------------------
with gr.Blocks(title="Kyutai TTS → MultiTalk (480 p)") as demo:
    gr.Markdown(
        "# Kyutai TTS → MultiTalk\n"
        "Two lines of dialogue + one reference image → talking-head clip (480 p, ≈15 s)."
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

    out_video   = gr.Video(label="Generated Clip")
    generate_btn = gr.Button("Generate")
    generate_btn.click(
        infer, [text_a, text_b, image_in, voice_a, voice_b], out_video
    )

if __name__ == "__main__":
    demo.queue(concurrency_count=1, max_size=5).launch()
