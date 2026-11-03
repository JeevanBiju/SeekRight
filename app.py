#!/usr/bin/env python3
"""
Gradio web app for SeekRight – automatic road-marking corner detection.

Usage:
    python app.py                  # local at http://127.0.0.1:7860
    python app.py --share          # + temporary public Gradio link

Deployment (Render / Railway / HF Spaces / any Docker host):
    The app auto-detects the PORT env var and binds to 0.0.0.0.
"""
import os
import sys

import cv2
import numpy as np

# ── make sure final.py (same folder) is importable ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final  # noqa: E402

import gradio as gr  # noqa: E402

# ── paths to sample images and their recommended colours ────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES = [
    {"path": os.path.join(BASE_DIR, "samples", "image_1.png"),  "color": "Yellow"},
    {"path": os.path.join(BASE_DIR, "samples", "test4.jpeg"),    "color": "Yellow"},
    {"path": os.path.join(BASE_DIR, "samples", "test5.jpeg"),    "color": "White"},
]


# ── core processing function ────────────────────────────────────────────────
def process_image(image_path, color_choice):
    """
    Run corner detection on the uploaded / selected image.
    Returns (annotated_image_rgb | None, result_text).
    """
    if image_path is None:
        return None, "⚠️ Please upload an image or click one of the samples below."

    # Map UI colour choice → final.py argument
    color_map = {"White": "white", "Yellow": "yellow", "Auto": "auto"}
    color_arg = color_map.get(color_choice, "auto")

    try:
        corners, area, col_used, mask = final.detect_corners(image_path, color_arg)
    except Exception as e:
        return None, f"❌ Error during detection:\n```\n{e}\n```"

    if corners is None:
        return None, (
            "❌ **No strip set detected.**\n\n"
            "Try a different colour setting or a clearer image."
        )

    # Annotate and convert BGR → RGB for Gradio
    annotated_bgr = final.annotate(image_path, corners, area)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Build result summary
    corner_lines = "\n".join(
        f"  Corner {i}: ({x:.1f}, {y:.1f})"
        for i, (x, y) in enumerate(corners, 1)
    )
    summary = (
        f"✅ **Detection Successful!**\n\n"
        f"**Colour detected:** {col_used}\n\n"
        f"**Corners (x, y):**\n{corner_lines}\n\n"
        f"**Polygon area:** {area:,.2f} px²"
    )

    return annotated_rgb, summary


# ── gallery click handler ───────────────────────────────────────────────────
def on_gallery_select(evt: gr.SelectData):
    """When a sample image in the gallery is clicked, load it and set colour."""
    idx = evt.index
    s = SAMPLES[idx]
    return s["path"], s["color"]


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="SeekRight – Road Marking Corner Detector") as demo:
    gr.Markdown(
        "# 🛣️ SeekRight – Road Marking Corner Detector\n"
        "Upload an image of road markings (speed-breaker / rumble strips) "
        "and the tool will detect the **4 corner points** of the strip set.\n\n"
        "Pick a colour to restrict detection, or leave it on "
        "**Auto** to try both yellow and white."
    )

    with gr.Row():
        # ── Left column: inputs ─────────────────────────────────────────
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Image",
                type="filepath",
                height=350,
            )
            color_input = gr.Radio(
                choices=["Auto", "White", "Yellow"],
                value="Auto",
                label="Strip Colour",
            )
            detect_btn = gr.Button(
                "🔍 Detect Corners", variant="primary", size="lg"
            )

        # ── Right column: outputs ───────────────────────────────────────
        with gr.Column(scale=1):
            image_output = gr.Image(
                label="Annotated Result",
                type="numpy",
                height=350,
            )
            result_text = gr.Markdown(label="Details")

    detect_btn.click(
        fn=process_image,
        inputs=[image_input, color_input],
        outputs=[image_output, result_text],
    )

    # ── Sample images gallery ───────────────────────────────────────────
    gr.Markdown(
        "---\n### 📸 Sample Images\n"
        "Click any image to load it for detection."
    )

    gallery = gr.Gallery(
        value=[s["path"] for s in SAMPLES],
        columns=3,
        rows=1,
        object_fit="contain",
        interactive=False,
        show_label=False,
        allow_preview=False,
    )

    # Clicking a gallery image → loads it into input + sets colour
    gallery.select(
        fn=on_gallery_select,
        inputs=[],
        outputs=[image_input, color_input],
    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = ap.parse_args()

    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=args.share,
        theme=gr.themes.Soft(),
    )
