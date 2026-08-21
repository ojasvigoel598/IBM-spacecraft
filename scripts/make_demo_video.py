"""Build the MissionMind demo video from LIVE-captured frames.

Pipeline:
  1. Generate narration segments with edge-tts (Microsoft neural voices, free,
     no API key). Each segment is a separate mp3 so moviepy can size each
     scene to exactly the length of its narration.
  2. Render PIL title cards (dark mission-control theme).
  3. Assemble with moviepy: static frames + hard cuts + narration track.
  4. Post-process with FFmpeg (bundled with imageio-ffmpeg, free): burn the
     narration script in as subtitles (SRT) and normalize loudness
     (EBU R128 loudnorm) so the track plays at a consistent level.

Output: demo/missionmind_demo.mp4  (1440x900, H.264, ~30fps) + demo/captions.srt
         Target duration: ≤ 3 minutes (IBM hackathon limit)
         Pitch: mission reliability + operator decision-making

Run:  .venv/Scripts/python.exe scripts/make_demo_video.py
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRAMES = os.path.join(ROOT, "demo", "frames")
AUDIO = os.path.join(ROOT, "demo", "audio")
TITLE_CARDS = os.path.join(ROOT, "demo", "cards")
OUT_MP4 = os.path.join(ROOT, "demo", "missionmind_demo.mp4")

VOICE = "en-US-ChristopherNeural"  # confident, natural, documentary feel
FPS = 30
W, H = 1440, 900

# ---- narration script: 10 scenes, mission-reliability pitch, ≤ 180 s ----
SCENES = [
    ("card_intro", "missionmind-intro",
     "What happens when a spacecraft fault occurs at three A M? The operator "
     "has minutes to decide. MissionMind gives them the answer, thirteen "
     "minutes before failure.",
     "MissionMind — fault detection, thirteen minutes early"),
    ("01_normal", "mission-control",
     "This is Mission Control. Every number comes from a live physics "
     "simulation. Solar: five hundred twenty watts. Battery full. "
     "Temperatures stable. The operator sees nominal, until the fault begins.",
     "Live physics simulation"),
    ("02_solar_fault", "fault-injection",
     "A solar array starts degrading. Power falls toward two hundred fifty "
     "watts. Net power goes negative. The battery begins draining. Without "
     "early detection, the operator loses the mission.",
     "Solar array degradation"),
    ("03_solar_fault_onset", "ml-detection",
     "At thirteen minutes, before the fault fully develops, the machine "
     "learning ensemble flags the anomaly. Zero false alarms during normal "
     "operations. The operator has time to act.",
     "ML ensemble detects the anomaly"),
    ("04_solar_deep", "rul-lead",
     "Remaining useful life: ninety six minutes. Not a guess, a "
     "physics-grounded prediction. The operator knows exactly how long "
     "they have to respond.",
     "RUL: physics-grounded prediction"),
    ("06_rag_evidence", "rag-evidence",
     "Why did this happen? The system retrieves the relevant engineering "
     "documentation, shows actual passages with relevance scores. The "
     "diagnosis is grounded in evidence, not guessed.",
     "Evidence-based diagnosis"),
    ("07_granite", "granite",
     "IBM watsonx Granite formats the assessment: risk level, probable "
     "cause, recommended action. Every claim traced to a source document. "
     "Not a chatbot paragraph, an engineering report.",
     "IBM watsonx Granite reasoning"),
    ("08_scenarios", "scenario-compare",
     "Solar degradation drains the battery. Radiator degradation overheats "
     "the bus. The physics makes the distinction clear. Two failure modes, "
     "two different operator responses.",
     "Failure-mode discrimination"),
    ("10_threejs", "digital-twin",
     "A Three.js digital twin responds in real time. Solar arrays dim on "
     "P V failure. Operators see the asset, not just numbers.",
     "3D digital twin, driven by live telemetry"),
    ("card_close", "missionmind-close",
     "MissionMind. Detected thirteen minutes early. Zero false alarms. "
     "Validated on real N A S A data. Clone the repository and run one "
     "command to start.",
     "MissionMind — one command to launch"),
]

TITLE_FONT = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arialbd.ttf")
BODY_FONT = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf")


# --------------------------------------------------------------------------
def build_srt(scenes: list, timings: list, out_path: str) -> None:
    """Write captions.srt from the narration script, one cue per scene.

    `timings` is a list of (key, duration_s) in the same order as `scenes`.
    Each scene's narration text is wrapped and shown for the scene window.
    """
    lines_out = []
    cue = 0
    t = 0.0
    for (key, _label, text, _title), (k2, dur) in zip(scenes, timings):
        assert key == k2
        wrapped = textwrap.wrap(text, width=72) or [""]
        start = t + 0.35
        end = t + dur - 0.15
        body = "\n".join(wrapped)
        cue += 1
        lines_out.append(f"{cue}\n{srt_ts(start)} --> {srt_ts(end)}\n{body}\n")
        t += dur
    with open(out_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines_out))
    print(f"SRT -> {os.path.relpath(out_path, ROOT)} ({cue} cues)")


def srt_ts(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def make_title_card(text: str, subtitle: str, path: str) -> None:
    """Dark mission-control title card (1440x900)."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (W, H), (7, 12, 22))
    d = ImageDraw.Draw(img)
    # starfield-ish noise dots
    import random
    rng = random.Random(7)
    for _ in range(260):
        x, y = rng.randrange(W), rng.randrange(H)
        b = rng.randrange(60, 160)
        d.ellipse([x, y, x + 2, y + 2], fill=(b, b + 20, b + 50))
    # accent line + satellite glyph
    d.rectangle([W // 2 - 220, 330, W // 2 + 220, 336], fill=(60, 120, 220))
    f_title = ImageFont.truetype(TITLE_FONT, 72) if os.path.exists(TITLE_FONT) else ImageFont.load_default()
    f_sub = ImageFont.truetype(BODY_FONT, 34) if os.path.exists(BODY_FONT) else ImageFont.load_default()
    lines = [text[i:i + 34] for i in range(0, len(text), 34)] or [text]
    y = 420
    for ln in lines:
        w_t = d.textlength(ln, font=f_title)
        d.text(((W - w_t) / 2, y), ln, fill=(235, 240, 250), font=f_title)
        y += 92
    if subtitle:
        w_s = d.textlength(subtitle, font=f_sub)
        d.text(((W - w_s) / 2, y + 20), subtitle, fill=(140, 180, 230), font=f_sub)
    d.text((40, H - 60), "MISSIONMIND · IBM Space Exploration AI", fill=(90, 110, 140),
           font=ImageFont.truetype(BODY_FONT, 22) if os.path.exists(BODY_FONT) else ImageFont.load_default())
    img.save(path)
    print("card", os.path.basename(path))


async def gen_audio() -> None:
    """Generate one mp3 per scene with edge-tts."""
    import edge_tts
    os.makedirs(AUDIO, exist_ok=True)
    for key, _label, text, _title in SCENES:
        out = os.path.join(AUDIO, f"{key}.mp3")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        comm = edge_tts.Communicate(text, VOICE, rate="+4%")
        await comm.save(out)
        print("audio", os.path.basename(out))
    print("audio done")


def main() -> int:
    try:
        asyncio.run(gen_audio())
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: edge-tts narration failed: {e}\n"
              f"Check network access to Microsoft TTS endpoints.")
        return 1

    os.makedirs(TITLE_CARDS, exist_ok=True)

    from moviepy import (AudioFileClip, ImageClip, concatenate_videoclips)

    clips = []
    timings = []  # (key, dur_s) in scene order, drives the SRT
    for key, label, text, title in SCENES:
        audio_path = os.path.join(AUDIO, f"{key}.mp3")
        if not os.path.exists(audio_path):
            print(f"skip {key}: no audio")
            continue
        audio = AudioFileClip(audio_path)
        dur = audio.duration + 0.9  # small tail so narration never clips
        timings.append((key, dur))

        if key.startswith("card_"):
            card_path = os.path.join(TITLE_CARDS, f"{key}.png")
            make_title_card(title.split(" — ")[0] if " — " in title else title,
                            title.split(" — ")[1] if " — " in title else "",
                            card_path)
            base = ImageClip(card_path).with_position("center")
        else:
            frame = os.path.join(FRAMES, f"{key}.png")
            base = ImageClip(frame).with_position("center")

        clip = base.with_duration(dur).with_audio(audio)
        clips.append(clip)

    # Hard cuts between scenes: narration carries the pacing, and skipping the
    # crossfade mask pipeline keeps memory usage flat (no compositing at all).
    video = concatenate_videoclips(clips, method="chain")
    video = video.with_fps(FPS)

    raw_mp4 = os.path.join(ROOT, "demo", "missionmind_demo_raw.mp4")
    video.write_videofile(
        raw_mp4, codec="libx264", audio_codec="aac",
        fps=FPS, preset="veryfast", bitrate="2500k",
        logger=None, threads=8, temp_audiofile=os.path.join(AUDIO, "_mix.m4a"),
        remove_temp=True,
    )
    print(f"RAW -> {raw_mp4} ({video.duration:.1f}s)")
    video.close()

    # FFmpeg post-pass (free, bundled binary): burn captions + loudnorm.
    srt_path = os.path.join(ROOT, "demo", "captions.srt")
    build_srt(SCENES, timings, srt_path)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    style = ("FontName=Arial,FontSize=24,PrimaryColour=&H00FFFFFF,"
             "OutlineColour=&H00101010,BorderStyle=1,Outline=2,Shadow=1,"
             "MarginV=42,Alignment=2")
    # Run from demo/ so the subtitles filter sees captions.srt relative.
    cmd = [ff, "-y", "-i", raw_mp4,
           "-vf", f"subtitles=captions.srt:force_style='{style}'",
           "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-b:a", "192k",
           os.path.basename(OUT_MP4)]
    print("FFmpeg: burning captions + loudnorm audio...")
    subprocess.run(cmd, cwd=os.path.join(ROOT, "demo"), check=True)
    os.remove(raw_mp4)
    print(f"VIDEO -> {OUT_MP4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
