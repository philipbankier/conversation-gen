import json, subprocess, uuid, os

LOW_VRAM_FLAGS = [
    "--num_persistent_param_in_dit", "0",
    "--use_teacache"
]

def generate_video(image_path: str, wav_paths, prompt: str) -> str:
    """Call MultiTalk CLI and return path to generated MP4."""
    job = {
        "image_path": image_path,
        "audio":      wav_paths,
        "prompt":     prompt
    }
    tmp_json = f"input_{uuid.uuid4().hex}.json"
    with open(tmp_json, "w") as f:
        json.dump(job, f)

    out_mp4 = f"out_{uuid.uuid4().hex}.mp4"
    cmd = [
        "python", "-m", "multitalk.generate_multitalk",
        "--ckpt_dir",  "Wan2.1-I2V-14B-480P",
        "--wav2vec_dir","chinese-wav2vec2-base",
        "--input_json", tmp_json,
        "--sample_steps","40",
        "--mode","streaming",
        *LOW_VRAM_FLAGS,
        "--save_file", out_mp4
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    os.remove(tmp_json)
    return out_mp4
