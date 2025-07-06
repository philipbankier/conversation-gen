---
title: Conversation-Gen(Kyutai-TTS → MultiTalk)
emoji: 🎭
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.27.0"
app_file: app.py
pinned: false
---

# Conversation Gen (Kyutai TTS → MultiTalk Space)

⚡ **One-click demo**: Two lines of dialogue + a reference image → talking-head clip (480 p, ≈15 s) with separate voices.

* **Kyutai TTS 1.6 B** — synthesize speech (≈200 preset voices).
* **Gemini-2.5-pro** — generates a single-sentence scene prompt.
* **MultiTalk 14 B (480 p)** — turns image + audio + prompt into MP4.

Runs on a single **L4 (24 GB)** GPU.

---

## Local quick-start

```bash
git clone <repo>
cd multitalk-kyutai-space
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=<your-gemini-key>
python app.py      # open http://127.0.0.1:7860
