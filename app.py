import streamlit as st
from PIL import Image
import numpy as np
import scipy.ndimage
import io
import re

from paddleocr import PaddleOCR
from streamlit_paste_button import paste_image_button as pbutton

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Text Reader",
    page_icon="\u270d",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ───────────────────────────────────────────── */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* ── Sidebar ──────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #7C9BF5;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }

    /* ── Section headers ──────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.6rem;
    }
    .section-header .bar {
        width: 3px;
        height: 20px;
        background: #7C9BF5;
        border-radius: 2px;
    }
    .section-header p {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8b949e;
        margin: 0;
    }

    /* ── Input card ───────────────────────────────────────── */
    .input-card {
        background: #161B22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    /* ── Output card ──────────────────────────────────────── */
    .output-card {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    .output-text {
        font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace;
        font-size: 0.85rem;
        line-height: 1.65;
        color: #E6EDF3;
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background: #0E1117;
        border: 1px solid #21262d;
        border-radius: 6px;
    }

    /* ── Stats bar ────────────────────────────────────────── */
    .stats-bar {
        display: flex;
        gap: 24px;
        padding: 0.5rem 0;
        font-size: 0.78rem;
        color: #8b949e;
    }
    .stats-bar .stat-val {
        color: #7C9BF5;
        font-weight: 600;
    }

    /* ── Copy button ──────────────────────────────────────── */
    .copy-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 6px;
        color: #E6EDF3;
        font-size: 0.8rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.15s;
        margin-top: 8px;
    }
    .copy-btn:hover { background: #30363d; border-color: #7C9BF5; }

    /* ── Image preview ────────────────────────────────────── */
    .img-preview {
        border: 1px solid #21262d;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .img-preview img {
        width: 100%;
        max-height: 320px;
        object-fit: contain;
        display: block;
        background: #0d1117;
    }

    /* ── Success / warning boxes ──────────────────────────── */
    .info-box {
        background: #161B22;
        border-left: 3px solid #7C9BF5;
        padding: 0.6rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #8b949e;
        margin: 0.5rem 0;
    }
    .warn-box {
        background: #1c1a0f;
        border-left: 3px solid #d29922;
        padding: 0.6rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #d29922;
        margin: 0.5rem 0;
    }

    /* ── Hide Streamlit branding ──────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# ── Session State ─────────────────────────────────────────────────────────────
if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = {}
if "ocr_words" not in st.session_state:
    st.session_state.ocr_words = []
if "image_dims" not in st.session_state:
    st.session_state.image_dims = None
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "Preserve Formatting"


# ── Cached OCR engine ────────────────────────────────────────────────────────
@st.cache_resource
def init_ocr(lang_code):
    return PaddleOCR(use_angle_cls=True, lang=lang_code, show_log=False)


# ── Image preprocessing ──────────────────────────────────────────────────────
def preprocess_image(img_rgb):
    """Grayscale -> denoise -> adaptive threshold -> deskew."""
    gray = np.array(img_rgb.convert("L"))

    denoised = scipy.ndimage.median_filter(gray, size=3).astype(np.float64)

    block_size = 11
    local_mean = scipy.ndimage.uniform_filter(denoised, size=block_size)
    thresh = np.where(denoised > local_mean - 2, 255, 0).astype(np.uint8)

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 0:
        mean = np.mean(coords, axis=0)
        centered = coords - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        if abs(angle) > 0.5:
            thresh = scipy.ndimage.rotate(
                thresh, angle, reshape=False, order=3,
                mode="constant", cval=0,
            ).astype(np.uint8)

    return thresh


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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### \u2699\ufe0f Settings")

    st.markdown("#### OCR Language")
    selected_langs = st.multiselect(
        "Languages",
        options=list(LANGUAGES.keys()),
        default=["English"],
        label_visibility="collapsed",
    )
    lang_code = LANGUAGES[selected_langs[0]] if selected_langs else "en"

    st.markdown("#### Confidence Threshold")
    conf_threshold = st.slider(
        "Minimum confidence (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
        help="Lines below this confidence score are discarded as noise.",
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem; color:#484f58; line-height:1.4;">'
        "UTF-8 byte-for-byte preservation<br>"
        "Every recognized character is kept intact"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main Layout ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><div class="bar"></div><p>Input</p></div>',
    unsafe_allow_html=True,
)

col_upload, col_paste = st.columns(2)

with col_upload:
    st.markdown(
        '<div class="input-card">',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif", "gif"],
        label_visibility="collapsed",
        help="Drag and drop or click to browse",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_paste:
    st.markdown(
        '<div class="input-card">',
        unsafe_allow_html=True,
    )
    paste_result = pbutton(
        label="Paste from Clipboard",
        background_color="#161B22",
        hover_background_color="#21262d",
        text_color="#E6EDF3",
        key="paste",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ── Determine image source ───────────────────────────────────────────────────
image_bytes = None

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
elif paste_result and paste_result.image_data is not None:
    buf = io.BytesIO()
    paste_result.image_data.save(buf, format="PNG")
    image_bytes = buf.getvalue()


# ── Process ───────────────────────────────────────────────────────────────────
if image_bytes:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        st.markdown(
            '<div class="warn-box">Could not read image. Ensure it is a valid image file.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    st.session_state.image_dims = img.size

    # Image preview (resized for display only)
    st.markdown(
        '<div class="section-header"><div class="bar"></div><p>Preview</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="img-preview">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    preprocessed = preprocess_image(img)

    # Run PaddleOCR on full-resolution preprocessed image
    with st.spinner("Running OCR..."):
        ocr_engine = init_ocr(lang_code)
        result = ocr_engine.ocr(preprocessed, cls=True)

    # Build word list — byte-for-byte preservation of PaddleOCR output
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

    st.session_state.ocr_words = words

    # Reconstruct all 3 modes
    st.session_state.ocr_results = {
        "Preserve Formatting": reconstruct_preserve(words),
        "Text + Whitespace Only": reconstruct_whitespace_only(words),
        "Clean Text": reconstruct_clean(words),
    }


# ── Output ────────────────────────────────────────────────────────────────────
if st.session_state.ocr_results:
    words = st.session_state.ocr_words

    st.markdown("---")
    st.markdown(
        '<div class="section-header"><div class="bar"></div><p>Output</p></div>',
        unsafe_allow_html=True,
    )

    # Interactive mode selector
    active_mode = st.radio(
        "Output Mode",
        options=["Preserve Formatting", "Text + Whitespace Only", "Clean Text"],
        horizontal=True,
        label_visibility="collapsed",
        key="mode_selector",
    )
    st.session_state.active_mode = active_mode

    result = st.session_state.ocr_results[active_mode]

    # Output text
    escaped_result = (
        result.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    st.markdown(
        f'<div class="output-card"><div class="output-text">{escaped_result}</div></div>',
        unsafe_allow_html=True,
    )

    # Copy button
    copy_js = f"""
    <script>
    function copyOutput() {{
        const text = {repr(result)};
        navigator.clipboard.writeText(text).then(function() {{
            const btn = document.getElementById('copy-btn');
            btn.innerHTML = '&#10003; Copied';
            btn.style.color = '#3fb950';
            btn.style.borderColor = '#3fb950';
            setTimeout(() => {{
                btn.innerHTML = '&#128203; Copy to Clipboard';
                btn.style.color = '#E6EDF3';
                btn.style.borderColor = '#30363d';
            }}, 2000);
        }});
    }}
    </script>
    <button class="copy-btn" id="copy-btn" onclick="copyOutput()">
        &#128203; Copy to Clipboard
    </button>
    """
    st.components.v1.html(copy_js, height=50)

    # Stats
    char_count = len(result)
    word_count = len(result.split())
    line_count = len(result.split("\n"))
    avg_conf = (
        round(sum(w["conf"] for w in words) / len(words), 1) if words else 0
    )

    st.markdown(
        f'<div class="stats-bar">'
        f'<span>Chars: <span class="stat-val">{char_count:,}</span></span>'
        f'<span>Words: <span class="stat-val">{word_count:,}</span></span>'
        f'<span>Lines: <span class="stat-val">{line_count:,}</span></span>'
        f"<span>Avg Confidence: <span class=\"stat-val\">{avg_conf}%</span></span>"
        f"<span>Lines Detected: <span class=\"stat-val\">{len(words):,}</span></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

else:
    # Landing state
    st.markdown(
        '<div class="info-box">'
        "Upload an image or paste one from your clipboard to begin. "
        "The OCR engine will extract all text with full UTF-8 character preservation."
        "</div>",
        unsafe_allow_html=True,
    )
