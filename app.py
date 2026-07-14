import cv2
import numpy as np
import re
import gradio as gr
from PIL import Image
from paddleocr import PaddleOCR


# ── Language Map (PaddleOCR 2.x codes) ──────────────────────────────────────
LANGUAGES = {
    "English": "en",
    "Chinese (Simplified)": "ch",
    "Chinese (Traditional)": "ch",
    "Japanese": "japan",
    "Korean": "korean",
    "Arabic": "ar",
    "French": "fr",
    "German": "german",
    "Spanish": "es",
    "Portuguese": "pt",
    "Italian": "it",
    "Russian": "ru",
    "Thai": "th",
    "Vietnamese": "vi",
}


# ── Cached OCR engine ────────────────────────────────────────────────────────
_ocr_cache = {}


def init_ocr(lang_code):
    if lang_code not in _ocr_cache:
        _ocr_cache[lang_code] = PaddleOCR(
            use_angle_cls=True, lang=lang_code, show_log=False
        )
    return _ocr_cache[lang_code]


# ── Image preprocessing ──────────────────────────────────────────────────────
def preprocess_image(img_rgb):
    """Grayscale -> denoise -> adaptive threshold -> deskew."""
    gray = cv2.cvtColor(np.array(img_rgb.convert("RGB")), cv2.COLOR_RGB2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            h, w = thresh.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            thresh = cv2.warpAffine(
                thresh, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


# ── Line / paragraph grouping ────────────────────────────────────────────────
def _group_into_lines(words, y_threshold=None):
    if not words:
        return []
    if y_threshold is None:
        heights = [w["h"] for w in words if w["h"] > 0]
        y_threshold = (sorted(heights)[len(heights) // 2] * 0.5) if heights else 10

    sorted_words = sorted(words, key=lambda w: (w["y"], w["x"]))
    lines = []
    current_line = [sorted_words[0]]

    for w in sorted_words[1:]:
        if abs(w["y"] - current_line[0]["y"]) < y_threshold:
            current_line.append(w)
        else:
            lines.append(sorted(current_line, key=lambda x: x["x"]))
            current_line = [w]
    lines.append(sorted(current_line, key=lambda x: x["x"]))
    return lines


def _group_into_paragraphs(lines, gap_multiplier=2.0):
    if not lines:
        return []
    heights = [max(w["h"] for w in line) for line in lines]
    avg_h = sum(heights) / len(heights) if heights else 20
    gap_threshold = avg_h * gap_multiplier

    paragraphs = [[lines[0]]]
    for i in range(1, len(lines)):
        prev_bottom = max(w["y"] + w["h"] for w in lines[i - 1])
        curr_top = min(w["y"] for w in lines[i])
        if (curr_top - prev_bottom) > gap_threshold:
            paragraphs.append([lines[i]])
        else:
            paragraphs[-1].append(lines[i])
    return paragraphs


# ── Reconstruction functions ──────────────────────────────────────────────────
def reconstruct_preserve(words):
    lines = _group_into_lines(words)
    paragraphs = _group_into_paragraphs(lines)
    parts = []
    for pi, para in enumerate(paragraphs):
        if pi > 0:
            parts.append("\n\n")
        for line in para:
            line_text = " ".join(w["text"] for w in line)
            parts.append(line_text)
            parts.append("\n")
    return "".join(parts).rstrip("\n")


def reconstruct_whitespace_only(words):
    lines = _group_into_lines(words)
    paragraphs = _group_into_paragraphs(lines)
    parts = []
    for pi, para in enumerate(paragraphs):
        if pi > 0:
            parts.append("\n\n")
        elif parts:
            parts.append("\n")
        for line in para:
            line_text = " ".join(w["text"] for w in line)
            parts.append(line_text)
    return "".join(parts)


def reconstruct_clean(words):
    lines = _group_into_lines(words)
    paragraphs = _group_into_paragraphs(lines)
    parts = []
    for pi, para in enumerate(paragraphs):
        if pi > 0:
            parts.append("\n\n")
        elif parts:
            parts.append("\n")
        for line in para:
            line_text = " ".join(w["text"] for w in line)
            parts.append(line_text.strip())
    raw = "".join(parts)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


# ── State for mode switching ─────────────────────────────────────────────────
_ocr_results = {"Preserve Formatting": "", "Text + Whitespace Only": "", "Clean Text": ""}


# ── Main OCR function ────────────────────────────────────────────────────────
def run_ocr(img, lang, confidence_threshold):
    if img is None:
        _ocr_results.update({"Preserve Formatting": "", "Text + Whitespace Only": "", "Clean Text": ""})
        return "", "Upload an image or paste one from your clipboard."

    lang_code = LANGUAGES.get(lang, "en")
    conf_threshold = confidence_threshold

    preprocessed = preprocess_image(img)

    ocr_engine = init_ocr(lang_code)
    result = ocr_engine.ocr(preprocessed, cls=True)

    words = []
    if result and result[0]:
        for line in result[0]:
            box = line[0]
            text = str(line[1][0])
            conf = float(line[1][1])

            x_min = min(p[0] for p in box)
            y_min = min(p[1] for p in box)
            x_max = max(p[0] for p in box)
            y_max = max(p[1] for p in box)

            if conf * 100 >= conf_threshold:
                words.append({
                    "text": text,
                    "x": int(x_min),
                    "y": int(y_min),
                    "w": int(x_max - x_min),
                    "h": int(y_max - y_min),
                    "conf": round(conf * 100, 1),
                })

    preserve = reconstruct_preserve(words)
    whitespace = reconstruct_whitespace_only(words)
    clean = reconstruct_clean(words)

    _ocr_results["Preserve Formatting"] = preserve
    _ocr_results["Text + Whitespace Only"] = whitespace
    _ocr_results["Clean Text"] = clean

    char_count = len(preserve)
    word_count = len(preserve.split())
    line_count = len(preserve.split("\n"))
    avg_conf = (
        round(sum(w["conf"] for w in words) / len(words), 1) if words else 0
    )
    stats = (
        f"**Chars:** {char_count:,} | **Words:** {word_count:,} | "
        f"**Lines:** {line_count:,} | **Avg Confidence:** {avg_conf}% | "
        f"**Detections:** {len(words):,}"
    )

    return preserve, stats


def switch_mode(mode):
    return _ocr_results.get(mode, "")


# ── Custom CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
#title { text-align: center; margin-bottom: 0.2rem; }
#subtitle { text-align: center; color: #8b949e !important; font-size: 0.85rem; margin-top: 0; margin-bottom: 0.5rem; }
footer { display: none !important; }
"""


# ── Theme ────────────────────────────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0E1117",
    body_background_fill_dark="#0E1117",
    block_background_fill="#161B22",
    block_background_fill_dark="#161B22",
    block_border_color="#21262d",
    block_border_color_dark="#21262d",
    block_label_text_color="#8b949e",
    block_label_text_color_dark="#8b949e",
    block_title_text_color="#E6EDF3",
    block_title_text_color_dark="#E6EDF3",
    input_background_fill="#0d1117",
    input_background_fill_dark="#0d1117",
    input_border_color="#21262d",
    input_border_color_dark="#21262d",
    button_primary_background_fill="#7C9BF5",
    button_primary_background_fill_dark="#7C9BF5",
    button_primary_text_color="#0E1117",
    button_primary_text_color_dark="#0E1117",
    button_secondary_background_fill="#21262d",
    button_secondary_background_fill_dark="#21262d",
    button_secondary_text_color="#E6EDF3",
    button_secondary_text_color_dark="#E6EDF3",
    text_color="#E6EDF3",
    text_color_dark="#E6EDF3",
    slider_color="#7C9BF5",
    slider_color_dark="#7C9BF5",
)


# ── Build Gradio UI ──────────────────────────────────────────────────────────
with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="Image Text Reader") as demo:
    gr.Markdown("# ✍️ Image Text Reader", elem_id="title")
    gr.Markdown(
        "Upload an image or paste from clipboard. OCR extracts all text with full UTF-8 character preservation.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            lang_dropdown = gr.Dropdown(
                choices=list(LANGUAGES.keys()),
                value="English",
                label="OCR Language",
            )
            conf_slider = gr.Slider(
                minimum=0,
                maximum=100,
                value=20,
                step=5,
                label="Minimum Confidence (%)",
                info="Lines below this score are discarded as noise.",
            )

        with gr.Column(scale=2):
            gr.Markdown("### 📥 Input")
            img_input = gr.Image(
                type="pil",
                label="Upload image (or paste from clipboard)",
                height=320,
            )

            gr.Markdown("### 📤 Output")
            mode_radio = gr.Radio(
                choices=["Preserve Formatting", "Text + Whitespace Only", "Clean Text"],
                value="Preserve Formatting",
                label="Output Mode",
            )
            output_text = gr.Textbox(
                label="Extracted Text",
                lines=16,
                show_copy_button=True,
                monospace=True,
                interactive=False,
            )
            stats_text = gr.Markdown()

    # Event handlers
    img_input.change(
        fn=run_ocr,
        inputs=[img_input, lang_dropdown, conf_slider],
        outputs=[output_text, stats_text],
    )
    lang_dropdown.change(
        fn=run_ocr,
        inputs=[img_input, lang_dropdown, conf_slider],
        outputs=[output_text, stats_text],
    )
    conf_slider.release(
        fn=run_ocr,
        inputs=[img_input, lang_dropdown, conf_slider],
        outputs=[output_text, stats_text],
    )
    mode_radio.change(
        fn=switch_mode,
        inputs=[mode_radio],
        outputs=[output_text],
    )


if __name__ == "__main__":
    demo.launch()
