"""Brighten captured frames for video using gamma correction + contrast lift.

Streamlit's dark theme resists CSS overrides, so we post-process the frames.
This produces much better video visibility than fighting the framework.

Run:  .venv/Scripts/python.exe scripts/brighten_frames.py [input_dir] [output_dir]
"""
import os
import sys
import numpy as np
from PIL import Image, ImageEnhance

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_DIR = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "demo", "bright_frames")
OUT_DIR = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(ROOT, "demo", "frames_bright")


def brighten_image(img, gamma=0.55, contrast=1.3, saturation=1.1):
    """Brighten a dark image using gamma correction + contrast boost.

    gamma < 1.0 brightens (0.55 lifts dark areas dramatically)
    contrast > 1.0 increases separation
    """
    arr = np.array(img, dtype=np.float32)

    # Gamma correction: linearize, apply gamma, re-encode
    arr = arr / 255.0
    arr = np.power(arr, gamma)  # gamma < 1 brightens
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    result = Image.fromarray(arr)

    # Contrast boost
    result = ImageEnhance.Contrast(result).enhance(contrast)

    # Slight saturation boost for color pop
    result = ImageEnhance.Color(result).enhance(saturation)

    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    count = 0
    for f in sorted(os.listdir(IN_DIR)):
        if not f.endswith(".png"):
            continue
        img = Image.open(os.path.join(IN_DIR, f))
        bright = brighten_image(img)
        out_path = os.path.join(OUT_DIR, f)
        bright.save(out_path)

        # Report brightness improvement
        orig_px = list(img.getdata())
        new_px = list(bright.getdata())
        orig_avg = sum(sum(p[:3]) / 3 for p in orig_px) / len(orig_px)
        new_avg = sum(sum(p[:3]) / 3 for p in new_px) / len(new_px)
        print(f"  {f}: {orig_avg:.0f}/255 -> {new_avg:.0f}/255 (+{new_avg - orig_avg:.0f})")
        count += 1

    print(f"\nBrightened {count} frames -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
