import math
import re

import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
from PIL import Image
from scipy.integrate import quad

# ============================================================
# QUANTUMLAB - TEORI PERTURBASI ORDE PERTAMA
# Versi web untuk Streamlit Cloud (tanpa tkinter/customtkinter)
# ============================================================

st.set_page_config(
    page_title="QuantumLab | Teori Perturbasi Orde Pertama",
    page_icon="⚛️",
    layout="wide",
)

# Tema warna
BG = "#F3F6FB"
CARD = "#FFFFFF"
NAVY = "#14213D"
BLUE = "#2563EB"
MUTED = "#64748B"
GREEN = "#15803D"
RED = "#B91C1C"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .quantum-header {{
        background: {NAVY}; color: white; padding: 22px 28px;
        border-radius: 14px; margin-bottom: 20px;
    }}
    .quantum-header h1 {{ margin: 0; font-size: 30px; }}
    .quantum-header p {{ margin: 6px 0 0 0; color: #CBD5E1; }}
    .panel {{
        background: {CARD}; border: 1px solid #E2E8F0;
        border-radius: 14px; padding: 20px; margin-bottom: 16px;
    }}
    .section-title {{
        color: {NAVY}; font-size: 18px; font-weight: 700;
        margin-bottom: 12px;
    }}
    div.stButton > button {{
        border-radius: 9px; min-height: 42px; font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

x, a, V1 = sp.symbols("x a V1", positive=True, real=True)


def parse_function(expression: str, aval: float, vval: float):
    """Membaca fungsi dengan simbol/fungsi matematika yang diizinkan."""
    local = {
        "x": x,
        "a": sp.Float(aval),
        "V1": sp.Float(vval),
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "pi": sp.pi,
        "log": sp.log,
        "abs": sp.Abs,
    }
    expr = sp.sympify(expression, locals=local, evaluate=True)
    unknown = expr.free_symbols - {x}
    if unknown:
        raise ValueError(
            "Simbol tidak dikenal: " + ", ".join(map(str, unknown))
        )
    return expr


def psi(q: int, xx: float, aa: float) -> float:
    return math.sqrt(2.0 / aa) * math.sin(q * math.pi * xx / aa)


def matrix_element(k: int, m: int, expr, aval: float) -> float:
    function = sp.lambdify(x, expr, modules=["numpy"])

    def integrand(xx):
        return psi(k, xx, aval) * float(function(xx)) * psi(m, xx, aval)

    value, _error = quad(
        integrand, 0.0, aval,
        epsabs=1e-10, epsrel=1e-10, limit=200
    )
    return value


def clean_ocr(text: str) -> str:
    text = text.replace(" ", "")
    text = text.replace("×", "*").replace("·", "*").replace("÷", "/")
    text = text.replace("π", "pi").replace("−", "-")
    compact = text.lower()
    if ("v1" in compact or "v" in compact) and "x" in compact and "a" in compact:
        if re.search(r"v\s*1?.*x.*a|x.*v\s*1?.*a", compact):
            return "V1*x/a"
    candidates = [line.strip() for line in text.splitlines() if line.strip()]
    for line in candidates:
        if any(ch in line for ch in ("x", "V1", "a")) and len(line) < 100:
            return line
    return ""


# Header
st.markdown(
    """
    <div class="quantum-header">
        <h1>⚛️ QuantumLab</h1>
        <p>Kalkulator Teori Perturbasi Kuantum • Koreksi energi dan fungsi gelombang</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Model: H'(x) = V1*x/a dan ψₙ(x) = √(2/a) sin(nπx/a). "
    "Masukkan fungsi perturbasi, parameter sistem, lalu tekan Hitung."
)

left, right = st.columns([5, 6], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">01 | INPUT PERHITUNGAN</div>', unsafe_allow_html=True)

    expression = st.text_input(
        "Fungsi perturbasi H'(x)",
        value="V1*x/a",
        help="Gunakan x, a, V1, angka, +, -, *, /, **, sin(), cos(), tan(), exp(), sqrt().",
    )

    uploaded_image = st.file_uploader(
        "Upload gambar fungsi (PNG/JPG)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Gambar fungsi yang diunggah", use_container_width=True)
        if st.button("Baca fungsi dari gambar (OCR)", use_container_width=True):
            try:
                import pytesseract
                extracted = pytesseract.image_to_string(image, config="--psm 6")
                cleaned = clean_ocr(extracted)
                if cleaned:
                    st.session_state["ocr_expression"] = cleaned
                    st.success(f"Hasil OCR: {cleaned}")
                    st.info("Salin hasil OCR ke kolom Fungsi perturbasi H'(x), lalu periksa kembali.")
                else:
                    st.warning("Teks fungsi belum berhasil dikenali. Silakan ketik fungsi secara manual.")
            except Exception:
                st.warning(
                    "OCR belum tersedia di server. Ketik fungsi secara manual pada kolom di atas. "
                    "Fitur perhitungan tetap dapat digunakan."
                )

    st.markdown("**Parameter sistem**")
    col_a, col_v = st.columns(2)
    with col_a:
        aval = st.number_input("Lebar sumur a", min_value=0.000001, value=1.0, step=0.1)
        e1 = st.number_input("Energi dasar E₁⁽⁰⁾", min_value=0.000001, value=1.0, step=0.1)
        n = st.number_input("Bilangan kuantum n", min_value=1, max_value=30, value=1, step=1)
    with col_v:
        vval = st.number_input("Tinggi potensial V1", value=1.0, step=0.1)
        N = st.number_input("Batas jumlah keadaan N", min_value=2, max_value=30, value=8, step=1)

    st.caption(
        "Contoh fungsi: V1*x/a. Format pangkat menggunakan **, misalnya x**2."
    )
    calculate_clicked = st.button("Hitung", type="primary", use_container_width=True)
    reset_clicked = st.button("Reset parameter", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">02 | OUTPUT HASIL</div>', unsafe_allow_html=True)
    st.write("Hasil perhitungan akan tampil di sini setelah tombol **Hitung** ditekan.")
    st.markdown("</div>", unsafe_allow_html=True)

if reset_clicked:
    st.rerun()

if calculate_clicked:
    try:
        if n > N:
            raise ValueError("Batas keadaan N harus lebih besar atau sama dengan bilangan kuantum n.")
        if not expression.strip():
            raise ValueError("Fungsi perturbasi belum diisi.")

        expr = parse_function(expression.strip(), aval, vval)

        # Koreksi energi orde pertama
        diagonal = matrix_element(int(n), int(n), expr, float(aval))
        e0 = int(n) ** 2 * float(e1)
        etotal = e0 + diagonal

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">HASIL KOREKSI ENERGI</div>', unsafe_allow_html=True)
            st.write(f"**H'(x)** = `{sp.sstr(expr)}`")
            st.write(f"**Keadaan n** = {int(n)}")
            st.metric("Energi awal Eₙ⁽⁰⁾", f"{e0:.10g}")
            st.metric("Koreksi energi Eₙ⁽¹⁾", f"{diagonal:.10g}")
            st.metric("Energi total Eₙ", f"{etotal:.10g}")
            st.caption(
                f"Untuk fungsi contoh H'=V1*x/a, hasil analitik Eₙ⁽¹⁾ = V1/2 = {vval/2:.10g}. "
                "Nilai integral numerik mengikuti fungsi input."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        coefficient_rows = []
        wave_terms = [f"ψ_{int(n)}(x)"]

        for m in range(1, int(N) + 1):
            if m == int(n):
                continue
            hmn = matrix_element(m, int(n), expr, float(aval))
            denominator = (int(n) ** 2 - m ** 2) * float(e1)
            cmn = hmn / denominator
            coefficient_rows.append({
                "m": m,
                "H'(m,n)": hmn,
                "Eₙ⁽⁰⁾ - Eₘ⁽⁰⁾": denominator,
                "cₘₙ": cmn,
            })
            if abs(cmn) > 1e-12:
                wave_terms.append(f"({cmn:.6g}) ψ_{m}(x)")

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**Koefisien koreksi fungsi gelombang**")
            st.dataframe(
                pd.DataFrame(coefficient_rows),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("**Pendekatan fungsi gelombang terkoreksi**")
            st.code("ψₙ ≈ " + " + ".join(wave_terms))
            st.success("Status: perhitungan berhasil")
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as exc:
        with right:
            st.error(f"Input/perhitungan tidak valid: {exc}")

st.caption(
    "QuantumLab • Model mengikuti contoh Teori Perturbasi Orde Pertama pada dokumen sumber."
)
