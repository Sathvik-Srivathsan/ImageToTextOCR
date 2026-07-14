import streamlit as st
import pytesseract
from PIL import Image
import io
import re

from paste_component import paste_image

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

    /* ── Mode pills ───────────────────────────────────────── */
    .mode-pills {
        display: flex;
        gap: 8px;
        margin-bottom: 0.5rem;
    }
    .mode-pill {
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        cursor: default;
        border: 1px solid #30363d;
        background: #161B22;
        color: #8b949e;
        transition: all 0.15s;
    }
    .mode-pill.active {
        background: #7C9BF5;
        color: #0E1117;
        border-color: #7C9BF5;
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

# ── Language Map ──────────────────────────────────────────────────────────────
LANGUAGES = {
    "English": "eng",
    "Chinese (Simplified)": "chi_sim",
    "Chinese (Traditional)": "chi_tra",
    "Japanese": "jpn",
    "Korean": "kor",
    "Arabic": "ara",
    "Hindi": "hin",
    "German": "deu",
    "French": "fra",
    "Spanish": "spa",
    "Portuguese": "por",
    "Italian": "ita",
    "Russian": "rus",
    "Thai": "tha",
    "Vietnamese": "vie",
}

# ── Session State ─────────────────────────────────────────────────────────────
if "ocr_result" not in st.session_state:
    st.session_state.ocr_result = None
if "ocr_words" not in st.session_state:
    st.session_state.ocr_words = []
if "image_dims" not in st.session_state:
    st.session_state.image_dims = None


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
    lang_code = "+".join(LANGUAGES[l] for l in selected_langs) if selected_langs else "eng"

    st.markdown("#### Confidence Threshold")
    conf_threshold = st.slider(
        "Minimum confidence",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        help="Words below this Tesseract confidence score are discarded as noise.",
    )

    st.markdown("#### Output Mode")
    output_mode = st.radio(
        "Mode",
        options=["Preserve Formatting", "Text + Whitespace Only", "Clean Text"],
        index=0,
        label_visibility="collapsed",
        help=(
            "Preserve Formatting: faithful spatial layout reproduction. "
            "Text + Whitespace Only: keeps _, -, \\u2014, \\n, \\t but removes position-based spacing. "
            "Clean Text: single spaces, single newlines, minimal output."
        ),
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem; color:#484f58; line-height:1.4;">'
        "UTF-8 byte-for-byte preservation<br>"
        "Every recognized character is kept intact"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Helper: reconstruct text from word list ──────────────────────────────────
def _median_word_width(words):
    if not words:
        return 10
    widths = [w["w"] for w in words if w["w"] > 0]
    return sorted(widths)[len(widths) // 2] if widths else 10


def _median_line_height(words):
    if not words:
        return 10
    heights = [w["h"] for w in words if w["h"] > 0]
    return sorted(heights)[len(heights) // 2] if heights else 10


def reconstruct_preserve(words):
    """Reconstruct text faithfully from spatial positions."""
    if not words:
        return ""

    med_w = _median_word_width(words)
    med_h = _median_line_height(words)

    lines_by_key = {}
    for w in words:
        key = (w["block"], w["par"], w["line"])
        lines_by_key.setdefault(key, []).append(w)

    sorted_lines = sorted(lines_by_key.keys())
    blocks_seen = set()
    paragraphs_seen = set()
    result_parts = []

    for line_key in sorted_lines:
        line_words = sorted(lines_by_key[line_key], key=lambda w: w["x"])
        block_key = (line_key[0],)
        par_key = (line_key[0], line_key[1])

        if block_key in blocks_seen:
            result_parts.append("\n")
        elif paragraphs_seen and par_key not in paragraphs_seen:
            result_parts.append("\n")
        blocks_seen.add(block_key)
        paragraphs_seen.add(par_key)

        if result_parts and result_parts[-1] != "\n" and result_parts[-1] != "\n\n":
            result_parts.append("\n")

        prev_end = None
        prev_y = None
        for w in line_words:
            if prev_end is not None:
                x_gap = w["x"] - prev_end
                y_gap = abs(w["y"] - prev_y) if prev_y is not None else 0

                if y_gap > med_h * 0.8:
                    result_parts.append("\n")
                    x_gap = w["x"]

                if x_gap > med_w * 3.5:
                    n_tabs = max(1, round(x_gap / (med_w * 4)))
                    result_parts.append("\t" * n_tabs)
                elif x_gap > med_w * 0.8:
                    result_parts.append(" ")
                else:
                    result_parts.append("")

            result_parts.append(w["text"])
            prev_end = w["x"] + w["w"]
            prev_y = w["y"]

    return "".join(result_parts)


def reconstruct_whitespace_only(words):
    """Keep UTF-8 text + semantic newlines, strip position-based spacing."""
    if not words:
        return ""

    lines_by_key = {}
    for w in words:
        key = (w["block"], w["par"], w["line"])
        lines_by_key.setdefault(key, []).append(w)

    sorted_lines = sorted(lines_by_key.keys())
    paragraphs_seen = set()
    result_parts = []

    for line_key in sorted_lines:
        line_words = sorted(lines_by_key[line_key], key=lambda w: w["x"])
        par_key = (line_key[0], line_key[1])

        if paragraphs_seen and par_key not in paragraphs_seen:
            result_parts.append("\n\n")
        elif result_parts:
            result_parts.append("\n")
        paragraphs_seen.add(par_key)

        for i, w in enumerate(line_words):
            if i > 0:
                result_parts.append(" ")
            result_parts.append(w["text"])

    return "".join(result_parts)


def reconstruct_clean(words):
    """Single spaces, single newlines, stripped fluff."""
    if not words:
        return ""

    lines_by_key = {}
    for w in words:
        key = (w["block"], w["par"], w["line"])
        lines_by_key.setdefault(key, []).append(w)

    sorted_lines = sorted(lines_by_key.keys())
    paragraphs_seen = set()
    result_parts = []

    for line_key in sorted_lines:
        line_words = sorted(lines_by_key[line_key], key=lambda w: w["x"])
        par_key = (line_key[0], line_key[1])

        if paragraphs_seen and par_key not in paragraphs_seen:
            result_parts.append("\n\n")
        elif result_parts:
            result_parts.append("\n")
        paragraphs_seen.add(par_key)

        line_text = " ".join(w["text"] for w in line_words)
        result_parts.append(line_text.strip())

    raw = "".join(result_parts)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


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
    st.markdown(
        '<div style="font-size:0.8rem; color:#8b949e; margin-bottom:6px;">'
        "Or paste from clipboard"
        "</div>",
        unsafe_allow_html=True,
    )
    pasted_data = paste_image(key="paste")
    st.markdown("</div>", unsafe_allow_html=True)


# ── Determine image source ───────────────────────────────────────────────────
image_bytes = None
image_source = None

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image_source = "upload"
elif pasted_data:
    import base64

    header, encoded = pasted_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image_source = "paste"


# ── Process ───────────────────────────────────────────────────────────────────
if image_bytes:
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        st.markdown(
            '<div class="warn-box">Could not read image. Ensure it is a valid image file.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # Resize large images for faster OCR (cap at 4000px on longest side)
    max_dim = 4000
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img_display = img.resize(new_size, Image.LANCZOS)
    else:
        img_display = img

    st.session_state.image_dims = img_display.size

    # Image preview
    st.markdown(
        '<div class="section-header"><div class="bar"></div><p>Preview</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="img-preview">', unsafe_allow_html=True)
    st.image(img_display, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Run OCR
    with st.spinner("Running OCR..."):
        data = pytesseract.image_to_data(
            img_display,
            lang=lang_code,
            output_type=pytesseract.Output.DICT,
            config="--psm 3 --oem 3",
        )

    # Build word list — byte-for-byte preservation of Tesseract output
    words = []
    for i in range(len(data["text"])):
        text_val = data["text"][i]
        if text_val is None:
            text_val = ""
        text_val = str(text_val)
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else -1
        if text_val.strip() and conf >= conf_threshold:
            words.append(
                {
                    "text": text_val,
                    "x": int(data["left"][i]),
                    "y": int(data["top"][i]),
                    "w": int(data["width"][i]),
                    "h": int(data["height"][i]),
                    "block": int(data["block_num"][i]),
                    "par": int(data["par_num"][i]),
                    "line": int(data["line_num"][i]),
                    "word": int(data["word_num"][i]),
                    "conf": conf,
                }
            )

    st.session_state.ocr_words = words

    # Reconstruct based on mode
    if output_mode == "Preserve Formatting":
        result = reconstruct_preserve(words)
    elif output_mode == "Text + Whitespace Only":
        result = reconstruct_whitespace_only(words)
    else:
        result = reconstruct_clean(words)

    st.session_state.ocr_result = result


# ── Output ────────────────────────────────────────────────────────────────────
if st.session_state.ocr_result is not None:
    result = st.session_state.ocr_result
    words = st.session_state.ocr_words

    st.markdown("---")
    st.markdown(
        '<div class="section-header"><div class="bar"></div><p>Output</p></div>',
        unsafe_allow_html=True,
    )

    # Mode indicator pill
    mode_labels = {
        "Preserve Formatting": "Preserve Formatting",
        "Text + Whitespace Only": "Text + Whitespace",
        "Clean Text": "Clean Text",
    }
    pills_html = '<div class="mode-pills">'
    for m in ["Preserve Formatting", "Text + Whitespace Only", "Clean Text"]:
        active = "active" if output_mode == m else ""
        pills_html += f'<div class="mode-pill {active}">{mode_labels[m]}</div>'
    pills_html += "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)

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
        f"<span>Words Recognized: <span class=\"stat-val\">{len(words):,}</span></span>"
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
