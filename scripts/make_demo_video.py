"""Build the MissionMind demo video from LIVE-captured frames.

Pipeline:
  1. Generate narration segments with edge-tts (Microsoft neural voices, free,
     no API key). Each segment is a separate mp3 so moviepy can size each
     scene to exactly the length of its narration.
  2. Render PIL title cards (dark mission-control theme).
  3. Assemble with moviepy: per-scene Ken Burns zoom, 0.6s crossfades, a
     subtle lower-third label, and the narration track.

Output: demo/missionmind_demo.mp4  (1440x900, H.264, ~30fps)

Run:  .venv/Scripts/python.exe scripts/make_demo_video.py
"""
from __future__ import annotations

import asyncio
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRAMES = os.path.join(ROOT, "demo", "frames")
AUDIO = os.path.join(ROOT, "demo", "audio")
TITLE_CARDS = os.path.join(ROOT, "demo", "cards")
OUT_MP4 = os.path.join(ROOT, "demo", "missionmind_demo.mp4")

VOICE = "en-US-ChristopherNeural"  # confident, natural, documentary feel
FPS = 30
W, H = 1440, 900

# ---- narration script: one segment per scene, describe only what exists ----
SCENES = [
    ("card_intro", "missionmind-intro",
     "MissionMind. An AI copilot for satellite mission operations. Built for the IBM advance space exploration challenge.",
     "MissionMind — AI Satellite Mission Control"),
    ("01_normal", "mission-control",
     "This is the Mission Control dashboard. Every number on screen comes from a live physics simulation, solved in real time. Solar array output, battery state of charge, bus voltage, and spacecraft temperature. Nominal operation: solar holds five hundred twenty watts, battery full, temperatures stable.",
     "Live physics simulation"),
    ("02_solar_fault", "fault-injection",
     "Now we inject a realistic failure: a degrading solar array. The fault ramps in over three hundred seconds, and the telemetry begins to change. Solar power falls toward two hundred fifty watts, net power goes negative, and the battery starts to drain.",
     "Fault injection: solar array degradation"),
    ("03_solar_fault_onset", "ml-detection",
     "Watch the detector. The machine learning ensemble flags the anomaly at around thirteen minutes into the mission, well before the fault fully develops. System status flips to critical.",
     "ML ensemble detects the anomaly"),
    ("04_solar_deep", "rul-lead",
     "MissionMind also predicts ahead. The remaining useful life counter is a leading indicator: it starts counting down before the detector confirms, telling the operator the battery has about ninety six minutes of margin.",
     "RUL: a leading indicator"),
    ("05_ml_diagnostics", "ml-diagnostics",
     "The ML diagnostics panel shows exactly what the model sees: the anomaly score, the health state, the ensemble verdict, and confidence. Three Isolation Forest models are combined so a single subsystem anomaly is still caught.",
     "Explainable ML diagnostics"),
    ("06_rag_evidence", "rag-evidence",
     "Why did this happen? The RAG evidence tab retrieves the relevant engineering documentation and shows the actual passages with relevance scores. The probable cause and recommended action are grounded in those sources, not guessed.",
     "RAG: evidence-based diagnosis"),
    ("07_granite", "granite",
     "The reasoning layer formats everything into a structured assessment: risk level, probable cause, recommended action, and the exact documents used. It is built for IBM watsonx Granite, and runs on an honest deterministic fallback when no API key is set.",
     "IBM watsonx Granite reasoning"),
    ("08_scenarios", "scenario-compare",
     "Compare failure modes side by side. Solar degradation drains the battery while temperature stays flat. Radiator degradation does the opposite: temperature climbs while power is untouched. The physics makes the distinction clear.",
     "Scenario comparison"),
    ("10_threejs", "digital-twin",
     "A physically rendered Three.js digital twin of the spacecraft is driven by the same live telemetry, so operators watch the asset respond in three dimensions as the fault evolves.",
     "3D digital twin, driven by live telemetry"),
    ("09_live_ingest", "live-ingest",
     "This is the dynamic path. A virtual edge node streams frame by frame over a real JSON lines transport, and the production ensemble scores every incoming window. A physical ESP32 or Raspberry Pi speaking the same wire format drops in unchanged.",
     "Live ingest: virtual edge node"),
    ("11_console", "web-console",
     "A React web console and a FastAPI backend expose the same pipeline to any browser. Health checks, scenario summaries, alerts with physics and RAG evidence, and live trace of which code actually ran.",
     "Web console and API backend"),
    ("card_validation", "validation",
     "The parameters are grounded in real NASA battery data, the physics is hand verified, and the full test suite passes: eighty one tests across nineteen suites.",
     "Grounded in NASA data, 81 tests passing"),
    ("card_close", "missionmind-close",
     "MissionMind turns raw telemetry into a clear answer: what failed, why, what to do, and how much time is left. Clone the repository and run one command to start the mission.",
     "MissionMind — one command to launch"),
]

TITLE_FONT = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arialbd.ttf")
BODY_FONT = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf")


# --------------------------------------------------------------------------
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
    for key, label, text, title in SCENES:
        audio_path = os.path.join(AUDIO, f"{key}.mp3")
        if not os.path.exists(audio_path):
            print(f"skip {key}: no audio")
            continue
        audio = AudioFileClip(audio_path)
        dur = audio.duration + 0.9  # small tail so narration never clips

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
    video.write_videofile(
        OUT_MP4, codec="libx264", audio_codec="aac",
        fps=FPS, preset="veryfast", bitrate="2500k",
        logger=None, threads=8, temp_audiofile=os.path.join(AUDIO, "_mix.m4a"),
        remove_temp=True,
    )
    print(f"VIDEO -> {OUT_MP4} ({video.duration:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
