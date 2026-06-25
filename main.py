#!/usr/bin/env python3
"""
Sale Deed Extractor for Indian Non-Judicial Stamp Paper PDFs
Extracts structured data from scanned Tamil/English sale deeds.

Usage:
    python extract_deed.py input.pdf
    python extract_deed.py file1.pdf file2.pdf -o results.json
    python extract_deed.py *.pdf -o batch_output.json --dpi 4.0

Options:
    --dpi FLOAT           Render zoom scale (default: 3.0 ≈ 216 DPI)
    --header-crop FLOAT   Fraction of page top to skip (default: 0.38)
    --lang STR            Tesseract language string (default: tam+eng)
    --psm INT             Tesseract page segmentation mode (default: 6)
    --oem INT             Tesseract OCR engine mode (default: 3)
    --config FILE         Load field config from a JSON file
    --output / -o FILE    Output JSON path (default: extracted_deeds.json)
    --csv                 Also write a CSV alongside the JSON
    --verbose             Print OCR text for debugging

Dependencies:
    pip install pymupdf pytesseract Pillow
    # Linux:  apt-get install tesseract-ocr tesseract-ocr-tam
    # macOS:  brew install tesseract tesseract-lang
"""

from __future__ import annotations

import sys
import re
import json
import csv
import argparse
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("deed_extractor")

# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("Missing dependency: pip install pymupdf")

try:
    import pytesseract
    from PIL import Image
except ImportError:
    sys.exit("Missing dependency: pip install pytesseract Pillow")


# ---------------------------------------------------------------------------
# Default configuration (can be overridden via --config JSON)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    # OCR settings
    "dpi_scale": 3.0,
    "header_crop": 0.38,
    "lang": "tam+eng",
    "psm": 6,
    "oem": 3,

    # ---- Amount patterns (tried in order; first match wins) ----
    "amount_patterns": [
        r"(?:ரூபாய்|Rs\.?)[^\d]*(\d[\d,]+)",
        r"(\d[\d,]+)\s*/\s*-",
        r"(\d[\d,]{4,})",          # bare number fallback (4+ digits)
    ],

    # ---- Date patterns ----
    # Named groups: day, month, year (numeric) OR year_word + month_word + day_word
    "date_numeric_pattern": r"(\d{1,2})[.\-/](\d{1,2})[.\-/]((19|20)\d{2})",
    "date_word_pattern":
        r"((19|20)\d{{2}}).*?({month_re}).*?(?:மாதம்)?.*?(\d{{1,2}})\s*[-–]\s*ம்\s*(?:தேதி|தெதி|தேதீ|தெதீ)",

    # Tamil month names → 2-digit month numbers
    "tamil_months": {
        "ஜனவரி":      "01",
        "பிப்ரவரி":   "02",
        "மார்ச்":     "03",
        "ஏப்ரல்":     "04",
        "மே":         "05",
        "ஜூன்":       "06",
        "ஜூலை":       "07",
        "ஆகஸ்ட்":    "08",
        "செப்டம்பர்": "09",
        "அக்டோபர்":  "10",
        "நவம்பர்":    "11",
        "டிசம்பர்":   "12",
    },

    # ---- Survey / plot patterns ----
    "survey_keyword_pattern": r"சர்வே\s+(\d+/\d+(?:\s*,\s*\d+/\d+)*)",
    "survey_fallback_pattern": r"(\d{2,4}/\d{1,4}(?:\s*,\s*\d{2,4}/\d{1,4})*)",

    "plot_patterns": [
        r"([A-Z]/\d+(?:/[A-Z0-9]+)?)",          # e.g. B/109, B/109/A
        r"(?:plot\s*no\.?|மனை\s*எண்\.?)\s*(\w+)",
    ],

    # ---- Extent / area patterns ----
    # Each entry: [pattern, unit_label]
    # Variants cover OCR noise: missing ் , swapped chars, space inside word
    "extent_patterns": [
        [r"(\d[\d,]*)\s*(?:சதுர\s*அடி|சமர\s*அடி|சதுர\s*அடிகள்|சதுரடிகள்|சதரடிகள்|sq\.?\s*ft)", "sq ft"],
        [r"(\d[\d,]*)\s*(?:cents?|சென்ட்|செண்ட்)",                        "cents"],
        [r"(\d[\d,]*)\s*(?:ஏக்கர்|ஏக்கார்|acres?)",                      "acres"],
        [r"(\d[\d,]*)\s*(?:grounds?|கிரவுண்ட்|கிராவுண்ட்)",              "grounds"],
    ],
    # Dimension: "80' அடியும் வடக்கு-தெற்கு 60' அடிகள் கொண்ட 4800"
    # OCR may drop the ' or use ` or " — also ஆடியும் for அடியும்
    "extent_dimension_pattern":
        r"(\d+)\s*[`'\"]\s*(?:அடியும்|ஆடியும்|அடி).*?(\d+)\s*[`'\"]\s*(?:அடிகள்|ஆடிகள்|அடி)",

    # ---- Buyer pattern ----
    # Capture text immediately before அவர்களுக்கு / அவர்களுக்கு
    "buyer_pattern": r"([^\n,]{3,80}?)\s+அவர்க(?:ளுக்கு|ள்|குக்கு)",

    # ---- Seller patterns ----
    "seller_patterns": [
        # Named near வசிக்கும்
        r"வசிக்கும்\s+(.{5,150}?)\s+(?:குமாரர்|சம்மதித்த|எழுதிக்கொடுத்த|பிள்ளை\s+ஆகிய)",
        # English-style signature
        r"(?:[A-Z]\.\s*)?[A-Z][a-z]+(?:h?a(?:si|da)(?:s|v)(?:an|am|ivan))",
    ],

    # ---- Output field ordering ----
    "output_fields": [
        "S.No", "Seller", "Buyer", "Amount", "Date",
        "Plot", "Survey No", "Extent", "Village", "Taluk", "District",
    ],
}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config_path:
        p = Path(config_path)
        if not p.exists():
            log.warning("Config file not found: %s — using defaults", config_path)
        else:
            with p.open(encoding="utf-8") as f:
                overrides = json.load(f)
            cfg.update(overrides)
            log.info("Loaded config overrides from %s", config_path)
    return cfg


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def pdf_to_pages_text(
    pdf_path: str,
    dpi_scale: float,
    header_crop: float,
    lang: str,
    psm: int,
    oem: int,
) -> list[str]:
    """
    Render each page, crop the decorative stamp-paper header, run Tesseract,
    and return a list of per-page text strings.
    """
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi_scale, dpi_scale)
    pages_text: list[str] = []

    tess_config = f"--psm {psm} --oem {oem}"

    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        w, h = img.size
        # Crop away the decorative stamp-paper header
        body = img.crop((0, int(h * header_crop), w, h))
        text = pytesseract.image_to_string(body, lang=lang, config=tess_config)
        pages_text.append(text)

    doc.close()
    return pages_text


def pages_to_full_text(pages: list[str]) -> str:
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Remove zero-width chars; collapse runs of spaces/blank lines."""
    for zw in ("\u200c", "\u200d", "\u200b", "\ufeff"):
        text = text.replace(zw, "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def extract_amount(text: str, cfg: dict) -> str:
    for pat in cfg["amount_patterns"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                return f"Rs. {int(raw):,}"
            except ValueError:
                return m.group(1)
    return ""


def extract_date(text: str, cfg: dict) -> str:
    # 1. Numeric dd-mm-yyyy / dd.mm.yyyy / dd/mm/yyyy
    m = re.search(cfg["date_numeric_pattern"], text)
    if m:
        return f"{m.group(1).zfill(2)}-{m.group(2).zfill(2)}-{m.group(3)}"

    month_map: dict[str, str] = cfg["tamil_months"]

    # Build a forgiving month regex: each key tried with optional ் at end
    # to handle OCR dropping the pulli mark (e.g. நவம்பர் → நவம்பர)
    month_variants = []
    for k in month_map:
        # strip trailing ் if present, make it optional
        base = k.rstrip("்")
        month_variants.append(re.escape(base) + "்?")
    month_re = "|".join(month_variants)

    # Mapping from stripped base → canonical key
    base_to_canonical = {k.rstrip("்"): k for k in month_map}

    def resolve_month(raw: str) -> str:
        raw = raw.strip()
        # Exact match first
        if raw in month_map:
            return month_map[raw]
        # Strip trailing ் and look up
        stripped = raw.rstrip("்")
        return month_map.get(base_to_canonical.get(stripped, ""), "??")

    # 2. Full Tamil word-form with day: "1989ம் வருடம் நவம்பர் மாதம் 1-ம் தேதி"
    #    Very tolerant — each part is optional except year and month.
    #    OCR dash variants: - – ‐ or even a space before ம்
    m = re.search(
        rf"((19|20)\d{{2}})"                              # group1+2: year
        rf".{{0,30}}?({month_re})"                        # group3: month (within 30 chars)
        rf".{{0,20}}?"                                    # anything between month and day
        rf"(\d{{1,2}})"                                   # group4: day number
        rf"\s*[-–‐]?\s*ம்"                               # -ம் (dash optional)
        rf".{{0,15}}?(?:தேதி|தெதி|தேதீ|தெதீ|தேதிய)?",  # தேதி optional
        text, re.DOTALL
    )
    if m:
        mm = resolve_month(m.group(3))
        day = m.group(4)
        # Sanity check: day must be 1-31
        if 1 <= int(day) <= 31:
            return f"{day.zfill(2)}-{mm}-{m.group(1)}"

    # 3. Looser still: year + month near மாதம் keyword, day anywhere after
    m = re.search(
        rf"((19|20)\d{{2}}).{{0,50}}?({month_re}).{{0,30}}?மாதம்.{{0,20}}?(\d{{1,2}})",
        text, re.DOTALL
    )
    if m:
        mm = resolve_month(m.group(3))
        day = m.group(4)
        if 1 <= int(day) <= 31:
            return f"{day.zfill(2)}-{mm}-{m.group(1)}"

    # 4. Year + month only (day not found)
    m = re.search(rf"((19|20)\d{{2}}).{{0,50}}?({month_re})", text, re.DOTALL)
    if m:
        mm = resolve_month(m.group(3))
        return f"??-{mm}-{m.group(1)}"

    return ""


def extract_survey(text: str, cfg: dict) -> str:
    m = re.search(cfg["survey_keyword_pattern"], text)
    if m:
        return m.group(1).replace(" ", "")
    m = re.search(cfg["survey_fallback_pattern"], text)
    if m:
        return m.group(1).replace(" ", "")
    return ""


def extract_plot(text: str, cfg: dict) -> str:
    for pat in cfg["plot_patterns"]:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            # Last occurrence in deed schedule is usually the subject plot
            return matches[-1]
    return ""


def extract_extent(text: str, cfg: dict) -> str:
    # 1. Explicit unit keyword (sq ft, cents, acres, grounds)
    for pat, unit in cfg["extent_patterns"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return f"{m.group(1).replace(',', '')} {unit}"

    # 2. Dimension product: "80' அடியும் ... 60' அடிகள்"
    #    OCR may drop the quote — try with and without quote char
    dim_pat = cfg["extent_dimension_pattern"]
    m = re.search(dim_pat, text, re.DOTALL)
    if m:
        try:
            area = int(m.group(1)) * int(m.group(2))
            return f"{area} sq ft"
        except (ValueError, IndexError):
            pass

    # 3. Dimension without quote chars: "80 அடியும் ... 60 அடிகள்"
    m = re.search(
        r"(\d+)\s*(?:அடியும்|ஆடியும்|அடி).{0,80}?(\d+)\s*(?:அடிகள்|ஆடிகள்|அடி)",
        text, re.DOTALL
    )
    if m:
        try:
            area = int(m.group(1)) * int(m.group(2))
            return f"{area} sq ft"
        except (ValueError, IndexError):
            pass

    # 4. Standalone area number immediately before/after காவிமனை keyword
    m = re.search(
        r"(\d{3,5})\s*(?:சதுரடிகள்|சதுர\s*அடி|காவிமனை)",
        text
    )
    if m:
        return f"{m.group(1)} sq ft"

    # 5. Area number that immediately follows கொண்ட (meaning "having/measuring")
    #    e.g. "கொண்ட 4800 சதுரடிகள்" or just "கொண்ட 4800"
    m = re.search(r"கொண்ட\s+(\d{3,5})", text)
    if m:
        return f"{m.group(1)} sq ft"

    return ""


def _clean_name(raw: str) -> str:
    raw = re.sub(r"\s+", " ", raw)
    raw = re.sub(r"^[^\w\u0B80-\u0BFF]+", "", raw)  # strip leading non-word (incl. Tamil)
    return raw.strip()


def extract_seller_buyer(text: str, cfg: dict) -> tuple[str, str]:
    buyer = seller = ""

    # Buyer — appears before அவர்களுக்கு
    m = re.search(cfg["buyer_pattern"], text)
    if m:
        raw = m.group(1)
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        buyer = _clean_name(parts[-1]) if parts else _clean_name(raw)

    # Seller — try each configured pattern
    for pat in cfg["seller_patterns"]:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            seller = _clean_name(m.group(0 if m.lastindex is None else 1)
                                  .replace("\n", " "))
            break

    return seller, buyer


# ---------------------------------------------------------------------------
# Location pattern tables
# Each entry: (canonical_display_name, [regex_variant, ...])
# Variants cover OCR noise: missing punctuation, swapped letters, partial reads
# ---------------------------------------------------------------------------

_VILLAGE_PATTERNS: list[tuple[str, list[str]]] = [
    ("காக்கூர்",        [r"காக்க[ூு]ர்?", r"காக்ஜ[ூு]ர்?", r"காக்க[ூு]",
                         r"[Kk]ak{1,2}[au][ur]", r"[Kk]akk?[aou]r", r"[Kk]ak[ck]ur"]),
    ("மயிலாப்பூர்",    [r"மய்?ி?லா[ப்]+ூர்?", r"[Mm]ylapore", r"[Mm]ailap"]),
    ("அண்ணா நகர்",     [r"அண்ண[ா]\s*நகர்?", r"[Aa]nna\s*[Nn]agar"]),
    ("வேளச்சேரி",      [r"வேளச்?சேரி", r"[Vv]elachery"]),
    ("அம்பத்தூர்",     [r"அம்பத்?தூர்?", r"[Aa]mbattur"]),
    ("தாம்பரம்",       [r"தாம்பரம்?", r"[Tt]ambaram"]),
    ("பெரம்பூர்",      [r"பெரம்பூர்?", r"[Pp]erambur"]),
    ("கோயம்பேடு",      [r"கோயம்பேடு", r"[Kk]oyambedu"]),
    ("வண்டலூர்",       [r"வண்டலூர்?", r"[Vv]andalur"]),
    ("திருவொற்றியூர்", [r"திருவொற்றி?யூர்?", r"[Tt]iruvottiyur"]),
    ("பள்ளிக்கரணை",   [r"பள்ளிக்கரணை", r"[Pp]allikaranai"]),
    ("கிழக்கம்பாக்கம்",[r"கிழக்கம்பாக்கம்", r"[Kk]izhakkambakkam"]),
    ("நரிக்குறவர்",    [r"நரிக்குறவர்", r"[Nn]arikkurav"]),
    ("செம்பாக்கம்",    [r"செம்பாக்கம்?", r"[Ss]embakkam"]),
    ("நெம்மேலி",       [r"நெம்மேலி", r"[Nn]emmeli"]),
]

_TALUK_PATTERNS: list[tuple[str, list[str]]] = [
    ("திருவள்ளூர்",  [r"திருவள்ள[ூு]ர்?", r"[Tt]hiru?vallur", r"[Tt]iruvallur"]),
    ("செங்கற்பட்டு", [r"செங்கற்?பட்?ட[ுூ]", r"[Cc]hengal?p[ae]t(?:tu)?"]),
    ("சென்னை",       [r"சென்னை", r"[Cc]hennai"]),
    ("காஞ்சிபுரம்",  [r"காஞ்சி(?:புரம்)?", r"[Kk]anchipuram"]),
    ("வேலூர்",       [r"வேலூர்?", r"[Vv]ell?ore?"]),
    ("திருவண்ணாமலை",[r"திருவண்ண[ா]மலை", r"[Tt]hiruvannamalai"]),
    ("விழுப்புரம்",  [r"விழுப்புரம்?", r"[Vv]illupuram"]),
    ("கடலூர்",       [r"கடலூர்?", r"[Cc]uddalore", r"[Kk]adalur"]),
    ("தாம்பரம்",     [r"தாம்பரம்?", r"[Tt]ambaram"]),
    ("மதுரை",        [r"மதுரை", r"[Mm]adurai"]),
    ("கோயம்புத்தூர்",[r"கோயம்புத்?தூர்?", r"[Cc]oimbatore"]),
    ("சேலம்",        [r"சேலம்?", r"[Ss]alem"]),
]

_DISTRICT_PATTERNS: list[tuple[str, list[str]]] = [
    ("செங்கற்பட்டு", [r"செங்கற்?பட்?ட[ுூ]", r"[Cc]hengal?p[ae]t(?:tu)?"]),
    ("திருவள்ளூர்",  [r"திருவள்ள[ூு]ர்?", r"[Tt]hiru?vallur", r"[Tt]iruvallur"]),
    ("சென்னை",       [r"சென்னை", r"[Cc]hennai"]),
    ("காஞ்சிபுரம்",  [r"காஞ்சி(?:புரம்)?", r"[Kk]anchipuram"]),
    ("வேலூர்",       [r"வேலூர்?", r"[Vv]ell?ore?"]),
    ("விழுப்புரம்",  [r"விழுப்புரம்?", r"[Vv]illupuram"]),
    ("கடலூர்",       [r"கடலூர்?", r"[Cc]uddalore"]),
    ("திருவண்ணாமலை",[r"திருவண்ண[ா]மலை", r"[Tt]hiruvannamalai"]),
    ("மதுரை",        [r"மதுரை", r"[Mm]adurai"]),
    ("கோயம்புத்தூர்",[r"கோயம்புத்?தூர்?", r"[Cc]oimbatore"]),
    ("சேலம்",        [r"சேலம்?", r"[Ss]alem"]),
]


def _match_location_patterns(pattern_list: list[tuple[str, list[str]]], text: str) -> str:
    """Return canonical name of first pattern group that matches anywhere in text."""
    for canonical, variants in pattern_list:
        for variant in variants:
            if re.search(variant, text):
                return canonical
    return ""


def extract_location(text: str, cfg: dict) -> tuple[str, str, str]:
    # --- Village ---
    # 1. Label-based: word(s) before கிராமம் / கிராம் / Village
    #    OCR may drop ் producing கிராம for கிராமம்
    village = ""
    # 1a. Label: "<name> கிராமம்" — handle OCR variants of கிராமம்
    m = re.search(
        r"([\u0B80-\u0BFF]+(?:\s+[\u0B80-\u0BFF]+)?)\s+"
        r"(?:கிராமம்|கிராமத்|கிராம்|கிரா(?:மம்|மத்|ம்)?|[Vv]illage)",
        text
    )
    if m:
        village = _clean_name(m.group(1))
    # 1b. Gram Panchayat pattern: "75 நம்பர் காக்கூர் கிராமம்"
    if not village:
        m = re.search(
            r"\d+\s*(?:நம்பர்|எண்\.?|[Nn]o\.?)\s*([\u0B80-\u0BFF]+)\s*"
            r"(?:கிராமம்|கிராமத்|கிராம்|கிரா(?:மம்|மத்|ம்)?)",
            text
        )
        if m:
            village = _clean_name(m.group(1))
    # 1c. Panchayat label: "<name> பஞ்சாயத்து"
    if not village:
        m = re.search(
            r"([\u0B80-\u0BFF]+)\s+பஞ்சாயத்?(?:து)?",
            text
        )
        if m:
            village = _clean_name(m.group(1))
    # 2. Fallback: pattern table
    if not village:
        village = _match_location_patterns(_VILLAGE_PATTERNS, text)

    # --- Taluk ---
    # 1. Try label-based: word(s) before சப்டிஸ்ட்ரிக்ட் / Taluk / Sub-District
    taluk = ""
    m = re.search(
        r"([\u0B80-\u0BFF\w]+(?:\s+[\u0B80-\u0BFF\w]+)?)\s+"
        r"(?:சப்டி(?:ஸ்ட்ரிக்ட்|லிட்)|[Tt]aluk|[Tt]aluq|[Ss]ub[- ]?[Dd]istrict)",
        text
    )
    if m:
        taluk = _clean_name(m.group(1))
    # 2. Fallback: pattern table
    if not taluk:
        taluk = _match_location_patterns(_TALUK_PATTERNS, text)

    # --- District ---
    # 1. Try label-based: word(s) before டிஸ்ட்ரிக்ட் / District
    district = ""
    m = re.search(
        r"([\u0B80-\u0BFF\w]+(?:\s+[\u0B80-\u0BFF\w]+)?)\s+"
        r"(?:டிஸ்ட்ரிக்ட்|[Dd]istrict)",
        text,
        re.IGNORECASE
    )
    if m:
        district = _clean_name(m.group(1))
    # 2. Fallback: pattern table
    if not district:
        district = _match_location_patterns(_DISTRICT_PATTERNS, text)

    return village, taluk, district


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def extract_deed_fields(
    full_text: str,
    cfg: dict,
    serial: int = 1,
) -> dict:
    text = normalize(full_text)

    amount                   = extract_amount(text, cfg)
    date                     = extract_date(text, cfg)
    survey                   = extract_survey(text, cfg)
    plot                     = extract_plot(text, cfg)
    extent                   = extract_extent(text, cfg)
    seller, buyer            = extract_seller_buyer(text, cfg)
    village, taluk, district = extract_location(text, cfg)

    return {
        "S.No":      serial,
        "Seller":    seller,
        "Buyer":     buyer,
        "Amount":    amount,
        "Date":      date,
        "Plot":      plot,
        "Survey No": survey,
        "Extent":    extent,
        "Village":   village,
        "Taluk":     taluk,
        "District":  district,
    }


def process_pdf(
    pdf_path: str,
    cfg: dict,
    serial: int = 1,
    verbose: bool = False,
) -> dict:
    log.info("[%d] OCR scanning: %s", serial, pdf_path)
    pages = pdf_to_pages_text(
        pdf_path,
        dpi_scale=cfg["dpi_scale"],
        header_crop=cfg["header_crop"],
        lang=cfg["lang"],
        psm=cfg["psm"],
        oem=cfg["oem"],
    )
    full_text = pages_to_full_text(pages)

    if verbose:
        print(f"\n{'='*60}\nOCR TEXT [{pdf_path}]\n{'='*60}")
        print(full_text)
        print("=" * 60)

    return extract_deed_fields(full_text, cfg, serial=serial)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(records: list[dict], out_path: Path, cfg: dict) -> None:
    fields = cfg["output_fields"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            # Flatten list fields for CSV
            flat = {
                k: (", ".join(v) if isinstance(v, list) else v)
                for k, v in rec.items()
            }
            writer.writerow(flat)
    log.info("CSV written → %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract structured fields from Indian sale-deed stamp-paper PDFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("pdfs", nargs="+", help="PDF file path(s) or glob pattern")
    p.add_argument("-o", "--output", default="extracted_deeds.json",
                   help="Output JSON file")
    p.add_argument("--csv", action="store_true",
                   help="Also write a CSV file alongside the JSON")
    p.add_argument("--config", default=None,
                   help="Path to a JSON config file to override defaults")
    p.add_argument("--dpi", type=float, default=None,
                   help="Render zoom scale (overrides config)")
    p.add_argument("--header-crop", type=float, default=None,
                   help="Fraction of page top to skip (overrides config)")
    p.add_argument("--lang", default=None,
                   help="Tesseract language string (overrides config)")
    p.add_argument("--psm", type=int, default=None,
                   help="Tesseract PSM mode (overrides config)")
    p.add_argument("--oem", type=int, default=None,
                   help="Tesseract OEM mode (overrides config)")
    p.add_argument("--verbose", action="store_true",
                   help="Print OCR text for each file (debug)")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = build_parser().parse_args()

    # Build config: defaults → JSON file → CLI flags
    cfg = load_config(args.config)
    if args.dpi         is not None: cfg["dpi_scale"]    = args.dpi
    if args.header_crop is not None: cfg["header_crop"]  = args.header_crop
    if args.lang        is not None: cfg["lang"]         = args.lang
    if args.psm         is not None: cfg["psm"]          = args.psm
    if args.oem         is not None: cfg["oem"]          = args.oem

    results: list[dict] = []
    for i, pdf_path in enumerate(args.pdfs, start=1):
        p = Path(pdf_path)
        if not p.exists():
            log.warning("Not found: %s — skipping", pdf_path)
            continue
        try:
            record = process_pdf(str(p), cfg, serial=i, verbose=args.verbose)
            results.append(record)
            log.info("Extracted: %s", json.dumps(record, ensure_ascii=False))
        except Exception as exc:
            log.error("Failed [%s]: %s", pdf_path, exc, exc_info=args.verbose)

    out = Path(args.output)
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Saved %d record(s) → %s", len(results), out.resolve())

    if args.csv:
        csv_path = out.with_suffix(".csv")
        write_csv(results, csv_path, cfg)


if __name__ == "__main__":
    main()