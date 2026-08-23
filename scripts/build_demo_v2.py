"""Build MissionMind hackathon demo v2 — judge-focused, 75-90 seconds.

Narrative structure:
  0-5s    HOOK       — 3D spacecraft fault animation (most visually impressive)
  5-15s   PROBLEM    — solar array degrading, power falling
  15-25s  DETECTION  — ML ensemble catches anomaly in 7 seconds
  25-45s  PIPELINE   — RAG evidence + Granite reasoning + 3D response
  45-60s  DEPTH      — physics simulation, RUL prediction, architecture
  60-75s  EVIDENCE   — NASA data, test counts, quantitative results
  75-85s  CLOSE      — one command to start, real product running

Output: demo/missionmind_demo_v2.mp4 (1600x900, H.264, ~30fps)
No hard-burned captions — clean video with optional SRT for accessibility.

Run:  .venv/Scripts/python.exe scripts/build_demo_v2.py
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRAMES = os.path.join(ROOT, "demo", "frames_bright")
AUDIO_DIR = os.path.join(ROOT, "demo", "audio_v2")
CARDS_DIR = os.path.join(ROOT, "demo", "cards_v2")
OUT_MP4 = os.path.join(ROOT, "demo", "missionmind_demo_v2.mp4")

VOICE = "en-US-ChristopherNeural"
FPS = 30
W, H = 1600, 900

# ---- narrative script: 90 seconds max, human-engineering tone ----
SCENES = [
    # (frame_key, audio_key, narration, card_title, card_subtitle)
    ("07_threejs", "hook",
     "Spacecraft faults happen at three A M. The operator has minutes "
     "to decide. MissionMind gives them the answer in seven seconds.",

     "MissionMind", "AI for spacecraft reliability"),

    ("01_normal", "problem",
     "Every number on this dashboard comes from a live physics simulation. "
     "Solar power, battery state, temperature. The operator sees nominal — "
     "until the fault begins.",

     "The Problem", "Spacecraft faults are hard to diagnose"),

    ("02_solar_fault", "fault_start",
     "A solar array starts degrading. Power falls toward two hundred fifty "
     "watts. Net power goes negative. The battery begins draining. Without "
     "early detection, the operator loses the mission.",

     "Solar Array Degradation", "Simulated power subsystem fault"),

    ("03_detection", "ml_detect",
     "Within seven seconds of fault onset, the machine learning ensemble "
     "flags the anomaly. Near zero false alarms during normal operations. "
     "The operator has thirty-nine minutes to act.",

     "ML Detection", "Ensemble of 8 anomaly detectors"),

    ("05_rag_evidence", "rag",
     "Why did this happen? The system retrieves relevant engineering "
     "documentation and shows actual passages with relevance scores. "
     "The diagnosis is grounded in evidence, not guessed.",

     "Evidence-Based Diagnosis", "TF-IDF retrieval over engineering docs"),

    ("06_granite", "granite",
     "IBM watsonx Granite formats the assessment: risk level, probable cause, "
     "recommended action. Every claim traced to a source document. "
     "Not a chatbot paragraph — an engineering report.",

     "IBM Granite Reasoning", "Structured output with citations"),

    ("04_deep_fault", "rul",
     "Remaining useful life: ninety-six minutes. Not a guess. A "
     "physics-grounded prediction with uncertainty bounds. "
     "The operator knows exactly how long they have to respond.",

     "RUL Prediction", "Physics-grounded with bootstrap CI"),

    ("07_threejs", "evidence",
     "Validated on real NASA battery data. AUC zero-point-seven-eight-six "
     "with six-seed robustness. Thirty test suites, all passing. "
     "One command to start the whole pipeline.",

     "NASA-Validated Results", "AUC 0.786 · 30 test suites · one command"),

    (None, "close",
     "MissionMind. Fault detection in seven seconds, thirty-nine minutes "
     "early. Clone the repository and run one command.",

     "MissionMind", "github.com/ojasvigoel598/IBM-spacecraft"),
]

TITLE_FONT = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arialbd.ttf")
BODY_FONT = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf")


def srt_ts(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(scenes, timings, out_path):
    """Write captions.srt from the narration script."""
    lines = []
    cue = 0
    t = 0.0
    for (_, _, text, _, _), (k, dur) in zip(scenes, timings):
        wrapped = textwrap.wrap(text, width=72) or [""]
        start = t + 0.3
        end = t + dur - 0.1
        body = "\n".join(wrapped)
        cue += 1
        lines.append(f"{cue}\n{srt_ts(start)} --> {srt_ts(end)}\n{body}\n")
        t += dur
    with open(out_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines))
    print(f"  SRT -> {os.path.relpath(out_path, ROOT)} ({cue} cues)")


def make_title_card(text, subtitle, path, bg=(13, 17, 23), accent=(0, 180, 255)):
    """Bright mission-control title card (1600x900)."""
    from PIL import Image, ImageDraw, ImageFont
    import random

    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    # subtle grid lines
    rng = random.Random(42)
    for y in range(0, H, 60):
        d.line([(0, y), (W, y)], fill=(bg[0] + 8, bg[1] + 12, bg[2] + 18), width=1)
    for x in range(0, W, 60):
        d.line([(x, 0), (x, H)], fill=(bg[0] + 5, bg[1] + 8, bg[2] + 12), width=1)

    # starfield dots
    for _ in range(80):
        x, y = rng.randrange(W), rng.randrange(H)
        b = rng.arange(40, 120) if hasattr(rng, "arange") else rng.randint(40, 120)
        d.ellipse([x, y, x + 2, y + 2], fill=(b, b + 15, b + 35))

    # accent line
    d.rectangle([W // 2 - 280, 340, W // 2 + 280, 346], fill=accent)

    # title
    f_title = ImageFont.truetype(TITLE_FONT, 78) if os.path.exists(TITLE_FONT) else ImageFont.load_default()
    f_sub = ImageFont.truetype(BODY_FONT, 30) if os.path.exists(BODY_FONT) else ImageFont.load_default()

    # center title
    bbox = d.textbbox((0, 0), text, font=f_title)
    tw = bbox[2] - bbox[0]
    d.text(((W - tw) // 2, 390), text, fill=(240, 245, 255), font=f_title)

    if subtitle:
        bbox2 = d.textbbox((0, 0), subtitle, font=f_sub)
        sw = bbox2[2] - bbox2[0]
        d.text(((W - sw) // 2, 490), subtitle, fill=(140, 180, 230), font=f_sub)

    # footer
    f_footer = ImageFont.truetype(BODY_FONT, 18) if os.path.exists(BODY_FONT) else ImageFont.load_default()
    d.text((40, H - 50), "MISSIONMIND · IBM Space Exploration AI",
           fill=(70, 90, 120), font=f_footer)

    img.save(path)
    print(f"  card {os.path.basename(path)}")


async def gen_audio():
    """Generate one mp3 per scene with edge-tts."""
    import edge_tts
    os.makedirs(AUDIO_DIR, exist_ok=True)
    for _, key, text, _, _ in SCENES:
        out = os.path.join(AUDIO_DIR, f"{key}.mp3")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        comm = edge_tts.Communicate(text, VOICE, rate="+3%")
        await comm.save(out)
        print(f"  audio {key}")
    print("  audio done")


def main():
    # 1. Generate narration audio
    print("=== Generating narration ===")
    try:
        asyncio.run(gen_audio())
    except Exception as e:
        print(f"FATAL: edge-tts failed: {e}")
        return 1

    # 2. Generate title cards
    print("\n=== Generating title cards ===")
    os.makedirs(CARDS_DIR, exist_ok=True)
    for _, key, _, title, subtitle in SCENES:
        card_path = os.path.join(CARDS_DIR, f"{key}.png")
        make_title_card(title, subtitle, card_path)

    # 3. Assemble video
    print("\n=== Assembling video ===")
    from moviepy import AudioFileClip, ImageClip, concatenate_videoclips

    clips = []
    timings = []

    for frame_key, audio_key, _, _, _ in SCENES:
        audio_path = os.path.join(AUDIO_DIR, f"{audio_key}.mp3")
        if not os.path.exists(audio_path):
            print(f"  skip {audio_key}: no audio")
            continue

        audio = AudioFileClip(audio_path)
        dur = audio.duration + 0.8  # tail padding
        timings.append((audio_key, dur))

        # Use frame from bright_frames or title card
        if frame_key and frame_key != "none":
            frame_path = os.path.join(FRAMES, f"{frame_key}.png")
        else:
            frame_path = None

        card_path = os.path.join(CARDS_DIR, f"{audio_key}.png")

        if frame_path and os.path.exists(frame_path):
            # Real dashboard frame — no card needed
            base = ImageClip(frame_path).with_position("center")
        elif os.path.exists(card_path):
            # Title card
            base = ImageClip(card_path).with_position("center")
        else:
            print(f"  skip {audio_key}: no frame or card")
            continue

        clip = base.with_duration(dur).with_audio(audio)
        clips.append(clip)

    if not clips:
        print("FATAL: no clips assembled")
        return 1

    video = concatenate_videoclips(clips, method="chain")
    video = video.with_fps(FPS)

    total_dur = video.duration
    print(f"\n  Total duration: {total_dur:.1f}s ({total_dur / 60:.1f} min)")
    if total_dur > 180:
        print(f"  WARNING: exceeds 3-minute limit!")

    # Write raw MP4
    raw_mp4 = os.path.join(ROOT, "demo", "v2_raw.mp4")
    video.write_videofile(
        raw_mp4, codec="libx264", audio_codec="aac",
        fps=FPS, preset="veryfast", bitrate="3000k",
        logger=None, threads=8,
        temp_audiofile=os.path.join(AUDIO_DIR, "_mix.m4a"),
        remove_temp=True,
    )
    print(f"  RAW -> {raw_mp4} ({video.duration:.1f}s)")
    video.close()

    # 4. FFmpeg post-pass: loudnorm audio only (NO subtitles burned in)
    print("\n=== FFmpeg post-processing ===")
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ff, "-y", "-i", raw_mp4,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        os.path.basename(OUT_MP4),
    ]
    print("  Normalizing audio...")
    subprocess.run(cmd, cwd=os.path.join(ROOT, "demo"), check=True)

    # Also write SRT for accessibility (NOT burned in)
    srt_path = os.path.join(ROOT, "demo", "captions_v2.srt")
    build_srt(SCENES, timings, srt_path)

    # Cleanup
    os.remove(raw_mp4)
    print(f"\n=== DONE ===")
    print(f"  VIDEO -> {OUT_MP4}")
    print(f"  SRT   -> {srt_path}")
    print(f"  Duration: {total_dur:.1f}s")
    print(f"  Resolution: {W}x{H}")
    print(f"  Format: H.264, ~30fps")
    print(f"  Captions: SRT file (not burned in)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
