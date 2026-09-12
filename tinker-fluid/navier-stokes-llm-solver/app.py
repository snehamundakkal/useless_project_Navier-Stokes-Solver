"""
app.py
======
🌊 LLM-Powered Navier–Stokes Problem Solver
Main Streamlit application.

Architecture:
    User Input → LLM (parameter extraction) → Python Solver (NumPy/SymPy)
    → Verification → LLM (explanation) → Display
"""

import os
import sys
import traceback
from typing import Optional

import streamlit as st
import numpy as np
from dotenv import load_dotenv

# ── ensure project root on path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

# ── Internal modules ──────────────────────────────────────────────────────────
from llm.client import (
    extract_problem_parameters,
    generate_explanation,
    is_demo_mode,
    OPENAI_MODEL,
)
from solver.analytical import (
    solve_plane_poiseuille,
    solve_couette,
    solve_pipe_flow,
    compute_reynolds,
    SolutionResult,
)
from solver.equations import GENERAL_NS_LATEX, CONTINUITY_LATEX, SIMPLIFICATION_STEPS
from solver.validation import run_verification, check_physical_validity
from visualization.plots import generate_plot
from examples.example_problems import EXAMPLE_PROBLEMS

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🌊 LLM Navier–Stokes Solver",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme management ──────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

def _get_css(theme: str) -> str:
    """Return full themed CSS ('dark' | 'light')."""
    is_dark = theme == "dark"
    if is_dark:
        rv = """
        --c-bg:#05080f;--c-bg2:#0a0f20;--c-bg3:#060c18;
        --c-sb:#06091a;--c-sb2:#0c1126;
        --c-card:rgba(11,17,38,.88);--c-card2:rgba(16,24,52,.92);
        --c-bdr:rgba(79,142,247,.18);--c-bdra:rgba(79,142,247,.55);
        --c-t:#dce8ff;--c-t2:#8892b0;--c-t3:#3d4d80;
        --c-a:#4f8ef7;--c-a2:#a78bfa;--c-a3:#f7c94f;
        --c-glow:rgba(79,142,247,.25);
        --c-met:rgba(12,20,50,.92);--c-inp:rgba(8,13,32,.85);
        --c-shd:0 10px 40px rgba(0,0,0,.6);
        --c-csh:0 4px 28px rgba(0,0,0,.45),0 0 1px rgba(79,142,247,.25);
        --c-hero:linear-gradient(135deg,#4f8ef7 0%,#a78bfa 50%,#f7c94f 100%);
        --c-pass:rgba(61,220,132,.1);--c-pt:#3ddc84;--c-pb:rgba(61,220,132,.4);
        --c-fail:rgba(247,84,84,.1);--c-ft:#f75454;--c-fb:rgba(247,84,84,.4);
        --c-demo:linear-gradient(90deg,rgba(247,196,79,.14),rgba(247,196,79,.04));
        --c-dt:#f7c94f;--c-db:rgba(247,196,79,.4);--c-da:#f7c94f;
        --c-lam:rgba(61,220,132,.16);--c-lt:#3ddc84;
        --c-tra:rgba(247,196,79,.16);--c-tt:#f7c94f;
        --c-tur:rgba(247,84,84,.16);--c-tut:#f75454;
        --c-lim:rgba(167,139,250,.08);--c-limt:#a78bfa;--c-limb:rgba(167,139,250,.28);
        --c-exp:rgba(6,11,28,.6);--c-code:#a5f3fc;
        --c-st:#06090f;--c-sth:#1a2550;
        --c-btn:linear-gradient(135deg,#3b6ff5,#5a3fcc);
        --c-bth:linear-gradient(135deg,#5a86ff,#7a5fff);
        --c-wb:rgba(247,196,79,.5);--c-wt:#f7c94f;--c-ib:rgba(247,84,84,.5);
        """
    else:
        rv = """
        --c-bg:#f0f5ff;--c-bg2:#e6effe;--c-bg3:#f4f8ff;
        --c-sb:#f7faff;--c-sb2:#ebf2ff;
        --c-card:rgba(255,255,255,.94);--c-card2:rgba(244,248,255,.97);
        --c-bdr:rgba(37,99,235,.16);--c-bdra:rgba(37,99,235,.50);
        --c-t:#0f172a;--c-t2:#475569;--c-t3:#94a3b8;
        --c-a:#2563eb;--c-a2:#7c3aed;--c-a3:#d97706;
        --c-glow:rgba(37,99,235,.18);
        --c-met:rgba(235,244,255,.96);--c-inp:rgba(255,255,255,.92);
        --c-shd:0 10px 40px rgba(37,99,235,.14);
        --c-csh:0 4px 22px rgba(37,99,235,.1),0 1px 4px rgba(0,0,0,.06);
        --c-hero:linear-gradient(135deg,#2563eb 0%,#7c3aed 50%,#d97706 100%);
        --c-pass:rgba(22,163,74,.1);--c-pt:#15803d;--c-pb:rgba(22,163,74,.38);
        --c-fail:rgba(220,38,38,.08);--c-ft:#dc2626;--c-fb:rgba(220,38,38,.32);
        --c-demo:linear-gradient(90deg,rgba(217,119,6,.1),rgba(217,119,6,.03));
        --c-dt:#b45309;--c-db:rgba(217,119,6,.32);--c-da:#d97706;
        --c-lam:rgba(22,163,74,.12);--c-lt:#15803d;
        --c-tra:rgba(217,119,6,.12);--c-tt:#b45309;
        --c-tur:rgba(220,38,38,.12);--c-tut:#dc2626;
        --c-lim:rgba(124,58,237,.07);--c-limt:#6d28d9;--c-limb:rgba(124,58,237,.22);
        --c-exp:rgba(248,251,255,.88);--c-code:#0369a1;
        --c-st:#dbeafe;--c-sth:#93c5fd;
        --c-btn:linear-gradient(135deg,#2563eb,#7c3aed);
        --c-bth:linear-gradient(135deg,#3b82f6,#8b5cf6);
        --c-wb:rgba(217,119,6,.42);--c-wt:#b45309;--c-ib:rgba(220,38,38,.42);
        """
    bg = "135deg,var(--c-bg) 0%,var(--c-bg2) 50%,var(--c-bg3) 100%"
    sb = "180deg,var(--c-sb) 0%,var(--c-sb2) 100%"
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{{{rv}}}
@keyframes gradientShift{{0%,100%{{background-position:0% 50%}}50%{{background-position:100% 50%}}}}
@keyframes floatWave{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-8px) rotate(5deg)}}}}
@keyframes slideUp{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes textGlow{{0%,100%{{background-position:0% 50%}}50%{{background-position:100% 50%}}}}
html,body,[class*="css"]{{font-family:'Plus Jakarta Sans','Inter',sans-serif!important}}
.stApp{{background:linear-gradient({bg});background-size:300% 300%;animation:gradientShift 22s ease infinite}}
section[data-testid="stSidebar"]{{background:linear-gradient({sb})!important;border-right:1px solid var(--c-bdr);backdrop-filter:blur(24px)}}
section[data-testid="stSidebar"] label{{color:var(--c-t)!important;font-weight:500}}
section[data-testid="stSidebar"] .stMarkdown p{{color:var(--c-t2)!important}}
.ns-card{{background:var(--c-card);border:1px solid var(--c-bdr);border-radius:20px;padding:24px 28px;margin:12px 0;box-shadow:var(--c-csh);backdrop-filter:blur(18px);animation:slideUp .45s ease;transition:transform .3s ease,box-shadow .3s ease,border-color .3s ease;color:var(--c-t)}}
.ns-card:hover{{transform:translateY(-3px);box-shadow:var(--c-shd);border-color:var(--c-bdra)}}
.ns-card-accent{{border-left:4px solid var(--c-a)}}
.hero-title{{font-family:'Plus Jakarta Sans',sans-serif!important;font-size:3.1rem;font-weight:800;background:var(--c-hero);background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.12;margin-bottom:.4rem;animation:textGlow 5s ease infinite;letter-spacing:-.8px}}
.hero-wave{{display:inline-block;animation:floatWave 3.2s ease-in-out infinite;-webkit-text-fill-color:initial}}
.hero-sub{{font-size:1.08rem;color:var(--c-t2);margin-bottom:2rem;font-style:italic;opacity:.92}}
.example-card{{background:var(--c-card);border:1px solid var(--c-bdr);border-radius:14px;padding:16px;cursor:pointer;transition:all .25s ease;backdrop-filter:blur(12px)}}
.example-card:hover{{border-color:var(--c-a);box-shadow:0 0 24px var(--c-glow);transform:translateY(-3px)}}
.example-title{{font-size:.95rem;font-weight:600;color:var(--c-t)}}
.example-desc{{font-size:.80rem;color:var(--c-t2);margin-top:4px}}
.assumption-badge{{display:inline-block;background:var(--c-pass);border:1px solid var(--c-pb);color:var(--c-pt);border-radius:20px;padding:4px 14px;margin:3px 4px;font-size:.82rem;font-weight:500;transition:transform .2s ease}}
.assumption-badge:hover{{transform:scale(1.06)}}
.flag-badge{{display:inline-block;background:var(--c-demo);border:1px solid var(--c-db);color:var(--c-dt);border-radius:8px;padding:4px 12px;margin:3px 0;font-size:.82rem;font-weight:500}}
.metric-box{{background:var(--c-met);border:1px solid var(--c-bdr);border-radius:18px;padding:20px 22px;text-align:center;margin:6px 2px;backdrop-filter:blur(12px);transition:transform .3s ease,box-shadow .3s ease;animation:slideUp .5s ease}}
.metric-box:hover{{transform:translateY(-4px);box-shadow:0 10px 28px var(--c-glow);border-color:var(--c-bdra)}}
.metric-label{{font-size:.73rem;color:var(--c-t2);text-transform:uppercase;letter-spacing:1.8px;font-weight:600;margin-bottom:8px}}
.metric-value{{font-size:1.65rem;font-weight:700;color:var(--c-a);font-family:'JetBrains Mono',monospace;line-height:1.2}}
.metric-unit{{font-size:.78rem;color:var(--c-t3);margin-top:5px}}
.verify-pass{{background:var(--c-pass);border:1px solid var(--c-pb);border-radius:14px;padding:16px 20px;color:var(--c-pt);animation:slideUp .4s ease}}
.verify-fail{{background:var(--c-fail);border:1px solid var(--c-fb);border-radius:14px;padding:16px 20px;color:var(--c-ft);animation:slideUp .4s ease}}
.demo-banner{{background:var(--c-demo);border:1px solid var(--c-db);border-left:4px solid var(--c-da);border-radius:12px;padding:12px 18px;margin-bottom:16px;color:var(--c-dt);font-size:.92rem;animation:slideUp .3s ease}}
.validity-valid{{background:var(--c-pass);border:1px solid var(--c-pb);border-radius:14px;padding:16px 20px;color:var(--c-pt);font-size:.92rem}}
.validity-warning{{background:var(--c-demo);border:1px solid var(--c-wb);border-left:4px solid var(--c-da);border-radius:14px;padding:16px 20px;color:var(--c-wt);font-size:.92rem}}
.validity-invalid{{background:var(--c-fail);border:1px solid var(--c-ib);border-left:4px solid var(--c-ft);border-radius:14px;padding:16px 20px;color:var(--c-ft);font-size:.92rem}}
.regime-badge-laminar{{background:var(--c-lam);border:1px solid var(--c-lt);color:var(--c-lt);border-radius:20px;padding:3px 14px;font-size:.82rem;font-weight:600}}
.regime-badge-transitional{{background:var(--c-tra);border:1px solid var(--c-tt);color:var(--c-tt);border-radius:20px;padding:3px 14px;font-size:.82rem;font-weight:600}}
.regime-badge-turbulent{{background:var(--c-tur);border:1px solid var(--c-tut);color:var(--c-tut);border-radius:20px;padding:3px 14px;font-size:.82rem;font-weight:600}}
.limitation-box{{background:var(--c-lim);border:1px solid var(--c-limb);border-radius:14px;padding:16px 20px;color:var(--c-limt);font-size:.88rem;margin-top:16px;animation:slideUp .45s ease}}
div[data-testid="stButton"]>button{{background:var(--c-btn)!important;color:white!important;border:none!important;border-radius:14px!important;padding:12px 32px!important;font-size:1rem!important;font-weight:700!important;letter-spacing:.4px!important;font-family:'Plus Jakarta Sans',sans-serif!important;transition:all .3s ease!important;width:100%!important;box-shadow:0 4px 18px var(--c-glow)!important}}
div[data-testid="stButton"]>button:hover{{background:var(--c-bth)!important;box-shadow:0 8px 30px var(--c-glow)!important;transform:translateY(-2px)!important}}
div[data-testid="stButton"]>button:active{{transform:translateY(0)!important}}
.stTextArea textarea{{background:var(--c-inp)!important;border:1px solid var(--c-bdr)!important;border-radius:14px!important;color:var(--c-t)!important;font-family:'Plus Jakarta Sans',sans-serif!important;font-size:.96rem!important;padding:14px 16px!important;transition:border-color .3s ease,box-shadow .3s ease!important;backdrop-filter:blur(12px)!important}}
.stTextArea textarea:focus{{border-color:var(--c-a)!important;box-shadow:0 0 0 3px var(--c-glow)!important}}
.stSelectbox>div>div{{background:var(--c-inp)!important;border:1px solid var(--c-bdr)!important;border-radius:10px!important;color:var(--c-t)!important}}
.streamlit-expanderHeader{{background:var(--c-card)!important;border-radius:14px!important;border:1px solid var(--c-bdr)!important;color:var(--c-t)!important;font-weight:600!important;transition:border-color .25s ease!important;padding:12px 16px!important}}
.streamlit-expanderHeader:hover{{border-color:var(--c-a)!important}}
.streamlit-expanderContent{{background:var(--c-exp)!important;border:1px solid var(--c-bdr)!important;border-top:none!important;border-radius:0 0 14px 14px!important;backdrop-filter:blur(12px)!important;padding:16px!important}}
hr{{border-color:var(--c-bdr)!important;opacity:.6;margin:1.2rem 0}}
.eq-scroll{{overflow-x:auto;max-width:100%}}
::-webkit-scrollbar{{width:6px;height:6px}}
::-webkit-scrollbar-track{{background:var(--c-st);border-radius:10px}}
::-webkit-scrollbar-thumb{{background:var(--c-sth);border-radius:10px}}
::-webkit-scrollbar-thumb:hover{{background:var(--c-a)}}
.stMarkdown p{{color:var(--c-t2);line-height:1.65}}
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4{{color:var(--c-t)!important;font-family:'Plus Jakarta Sans',sans-serif!important}}
.stMarkdown strong{{color:var(--c-t)!important}}
code,pre{{background:var(--c-met)!important;border:1px solid var(--c-bdr)!important;border-radius:8px!important;color:var(--c-code)!important;font-family:'JetBrains Mono',monospace!important}}
.stAlert{{border-radius:12px!important}}
.katex,.katex-display{{color:var(--c-t)!important}}
</style>"""


st.markdown(_get_css(st.session_state.theme), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metric_html(label: str, value: str, unit: str = "") -> str:
    return (
        f'<div class="metric-box">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-unit">{unit}</div>'
        f'</div>'
    )


def _assumption_badges(assumptions: list[str]) -> str:
    return " ".join(
        f'<span class="assumption-badge">✓ {a}</span>'
        for a in assumptions
    )


def _fmt(val: Optional[float], decimals: int = 6, sci: bool = False) -> str:
    if val is None:
        return "—"
    if sci or (abs(val) < 1e-3 and val != 0):
        return f"{val:.{decimals}e}"
    return f"{val:.{decimals}g}"


# ─────────────────────────────────────────────────────────────────────────────
# Solver dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_solver(params: dict) -> SolutionResult:
    """Route extracted parameters to the correct analytical solver."""
    ptype = params.get("problem_type", "unknown")
    mu = params.get("viscosity")
    rho = params.get("density") or 1000.0
    dpdx = params.get("pressure_gradient") or 0.0

    if ptype == "poiseuille":
        h = params.get("h")
        if h is None:
            return SolutionResult("poiseuille", False, "Plate separation 'h' is required.")
        if mu is None:
            mu = 0.001   # default water viscosity
        return solve_plane_poiseuille(h=h, mu=mu, dpdx=dpdx, rho=rho)

    elif ptype == "couette":
        h = params.get("h")
        U = params.get("velocity")
        if h is None:
            return SolutionResult("couette", False, "Plate separation 'h' is required.")
        if U is None:
            return SolutionResult("couette", False, "Upper-plate velocity 'U' is required.")
        if mu is None:
            mu = 0.001
        return solve_couette(h=h, U=U, mu=mu, rho=rho, dpdx=dpdx)

    elif ptype == "pipe":
        R = params.get("R")
        if R is None:
            return SolutionResult("pipe", False, "Pipe radius 'R' is required.")
        if mu is None:
            mu = 0.001
        return solve_pipe_flow(R=R, mu=mu, dpdx=dpdx, rho=rho)

    elif ptype == "reynolds":
        V = params.get("velocity")
        L = params.get("h") or params.get("R")
        if L is not None and params.get("R"):
            L = 2.0 * params["R"]   # diameter for pipe
        if V is None:
            return SolutionResult("reynolds", False, "Characteristic velocity is required.")
        if L is None:
            return SolutionResult("reynolds", False, "Characteristic length (h or R) is required.")
        if mu is None:
            mu = 0.001
        return compute_reynolds(rho=rho, velocity=V, length=L, mu=mu)

    else:
        return SolutionResult(
            "unknown", False,
            "This problem type is not currently supported. "
            "Supported types: Poiseuille flow, Couette flow, Pipe flow, Reynolds number."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Theme Toggle
    _dark = st.session_state.theme == "dark"
    st.markdown(
        ('<div style="text-align:center;padding:12px 0 8px;font-size:1rem;font-weight:700;">'
         + ('🌙 Dark Mode' if _dark else '☀️ Light Mode') + '</div>'),
        unsafe_allow_html=True,
    )
    _t1, _t2 = st.columns(2)
    with _t1:
        if st.button("🌙 Dark", key="btn_theme_dark", use_container_width=True):
            st.session_state.theme = "dark"
            st.rerun()
    with _t2:
        if st.button("☀️ Light", key="btn_theme_light", use_container_width=True):
            st.session_state.theme = "light"
            st.rerun()
    st.markdown("---")
    st.markdown("## ⚙️ Model Settings")
    st.markdown("---")

    demo_forced = st.toggle("🧪 Force Demo Mode", value=is_demo_mode(),
                             help="Run without an LLM API key using predefined responses.")
    if demo_forced:
        os.environ["DEMO_MODE"] = "true"
    else:
        if "DEMO_MODE" in os.environ:
            del os.environ["DEMO_MODE"]

    st.markdown("**LLM Provider**")
    provider = st.selectbox("Provider", ["OpenAI"], label_visibility="collapsed")

    model_choice = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="Select the OpenAI model to use.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.0, value=0.05, step=0.05,
        help="Lower = more deterministic. Recommended ≤ 0.1 for extraction.",
    )

    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown(
        """
        **Architecture:**
        ```
        User Input
            ↓
          LLM
        (extract params)
            ↓
        Python Solver
        (NumPy + SymPy)
            ↓
        Verification
            ↓
          LLM
        (explanation)
            ↓
          Display
        ```
        """
    )
    st.markdown("---")
    st.markdown(
        '<div class="limitation-box">'
        "⚠️ <b>Limitation Notice</b><br>"
        "This system solves selected Navier–Stokes special cases. "
        "It does <b>not</b> solve the general 3D Navier–Stokes existence "
        "and smoothness problem."
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Page Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="hero-title"><span class="hero-wave">🌊</span> LLM Navier–Stokes Solver</div>'
    '<div class="hero-sub">'
    "Ask a fluid mechanics problem. Let the LLM interpret it. Let mathematics verify it."
    "</div>",
    unsafe_allow_html=True,
)

if is_demo_mode() or demo_forced:
    st.markdown(
        '<div class="demo-banner">'
        "🧪 <b>DEMO MODE</b> — Running with predefined LLM responses. "
        "Add your OpenAI API key to <code>.env</code> for full LLM functionality."
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Example Problem Cards
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 🎯 Example Problems")
cols = st.columns(4)
selected_example = None

for i, ex in enumerate(EXAMPLE_PROBLEMS):
    with cols[i]:
        if st.button(
            f"{ex.icon} {ex.title}",
            key=f"btn_{ex.id}",
            help=ex.description,
        ):
            selected_example = ex

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Problem Input
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### ✍️ Enter Your Problem")

default_text = selected_example.problem_text if selected_example else ""
if "last_example" not in st.session_state:
    st.session_state.last_example = ""

# Update text area when example is selected
if selected_example and selected_example.problem_text != st.session_state.last_example:
    st.session_state.last_example = selected_example.problem_text
    st.session_state.problem_text = selected_example.problem_text

problem_text = st.text_area(
    "Describe your fluid problem",
    value=st.session_state.get("problem_text", ""),
    height=140,
    placeholder=(
        "e.g. Water with viscosity 0.001 Pa·s flows between two stationary parallel plates "
        "separated by 0.02 m. The pressure gradient is -100 Pa/m. Find the velocity profile."
    ),
    key="problem_input",
    label_visibility="collapsed",
)

st.markdown(
    "<small style='color:#5566aa'>💡 Tip: Include fluid properties (viscosity, density), "
    "geometry (plate spacing or pipe radius), and boundary conditions.</small>",
    unsafe_allow_html=True,
)
st.markdown("")

solve_clicked = st.button("🔍 SOLVE PROBLEM", key="solve_btn")


# ─────────────────────────────────────────────────────────────────────────────
# Main Solve Pipeline
# ─────────────────────────────────────────────────────────────────────────────

if solve_clicked and problem_text.strip():
    st.markdown("---")
    st.markdown("## 📊 Solution")

    # ── STEP 1: LLM parameter extraction ──────────────────────────────────────
    with st.spinner("🤖 LLM is analysing your problem…"):
        params, llm_err = extract_problem_parameters(
            problem_text,
            temperature=temperature,
            model=model_choice,
        )

    if llm_err:
        st.error(f"⚠️ LLM Error: {llm_err}")
        st.info("The solver is switching to Demo Mode for this problem.")
        from llm.client import get_demo_parsed_response, extract_parameters_for_solver as _eps
        demo_raw = get_demo_parsed_response(problem_text)
        params = _eps(demo_raw)

    if params is None:
        st.error("Could not extract parameters from the problem. Please try rephrasing.")
        st.stop()

    # ── STEP 2: Python Solver ─────────────────────────────────────────────────
    with st.spinner("🔢 Python solver running…"):
        try:
            result = dispatch_solver(params)
        except Exception as e:
            st.error(f"Solver error: {e}")
            st.code(traceback.format_exc())
            st.stop()

    # ── STEP 3: Verification ──────────────────────────────────────────────────
    verification = None
    physical_validity = None
    if result.success:
        with st.spinner("Running verification…"):
            try:
                # Mathematical: SymPy vs NumPy (purely algebraic)
                verification = run_verification(result)
                # Physical: Re vs laminar threshold (fully deterministic)
                physical_validity = check_physical_validity(result)
            except Exception as e:
                st.warning(f"Verification could not run: {e}")

    # ── STEP 4: Build re_info for LLM ────────────────────────────────────────
    # Deterministic Re + regime passed to LLM so it CANNOT invent the flow regime.
    re_info = None
    if physical_validity is not None:
        re_info = {
            "re_value": physical_validity.re_value,
            "flow_regime": physical_validity.flow_regime,
            "validity_label": physical_validity.validity_label,
            "re_formula_description": physical_validity.re_formula_description,
            "char_length_name": physical_validity.char_length_name,
            "re_limit_laminar": physical_validity.re_limit_laminar,
            "detail_message": physical_validity.detail_message,
        }
    elif result.reynolds_number is not None:
        re_info = {
            "re_value": result.reynolds_number,
            "flow_regime": ("Laminar" if result.reynolds_number < 2300
                           else "Transitional" if result.reynolds_number < 4000
                           else "Turbulent"),
            "validity_label": "N/A",
            "re_formula_description": result.re_formula_description or "",
            "char_length_name": result.re_char_length_name or "L",
            "re_limit_laminar": 2300.0,
            "detail_message": "",
        }

    # ── STEP 5: LLM Explanation ───────────────────────────────────────────────
    numerical_summary = {}
    if result.success:
        regime_str = physical_validity.flow_regime if physical_validity else "Unknown"
        numerical_summary = {
            "Problem type": result.problem_type,
            "Maximum velocity": f"{_fmt(result.u_max)} m/s",
            "Average velocity": f"{_fmt(result.u_avg)} m/s",
            "Flow rate": f"{_fmt(result.flow_rate)} m³/s (or m²/s for 2D)",
            "Reynolds number (deterministic)": (
                f"{_fmt(result.reynolds_number)} "
                f"[{result.re_formula_description or 'Re=ρVL/μ'}]"
            ),
            "Flow regime (deterministic)": regime_str,
            "Wall shear stress": f"{_fmt(result.wall_shear_stress)} Pa",
        }

    with st.spinner("LLM generating explanation…"):
        explanation, exp_err = generate_explanation(
            problem_text, params, numerical_summary,
            re_info=re_info,
            temperature=0.3, model=model_choice
        )
    if exp_err:
        st.warning(f"LLM explanation error (using fallback): {exp_err}")

    # ─────────────────────────────────────────────────────────────────────────
    # DISPLAY RESULTS
    # ─────────────────────────────────────────────────────────────────────────

    # 1. Problem Understanding
    with st.expander("1️⃣  Problem Understanding", expanded=True):
        st.markdown(
            f'<div class="ns-card ns-card-accent">'
            f'<b>Problem Type:</b> {params.get("problem_type", "unknown").title()}<br>'
            f'<b>Geometry:</b> {params.get("geometry", "—")}<br>'
            f'<b>Fluid:</b> {params.get("fluid", "—").title()}<br>'
            f'<b>Summary:</b> {params.get("problem_summary", "—")}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if params.get("flags"):
            st.markdown("**⚠️ Flags / Warnings:**")
            for f in params["flags"]:
                st.markdown(f'<div class="flag-badge">⚠ {f}</div>', unsafe_allow_html=True)

    # 2. Given Parameters
    with st.expander("2️⃣  Given Parameters", expanded=True):
        table_data = {
            "Parameter": ["Fluid", "Density (ρ)", "Dynamic Viscosity (μ)",
                          "Pressure Gradient (dp/dx)", "Velocity (U or V)",
                          "Plate Separation (h)", "Pipe Radius (R)"],
            "Value": [
                params.get("fluid", "—").title(),
                f"{params.get('density', '—')} kg/m³",
                f"{params.get('viscosity', '—')} Pa·s",
                f"{params.get('pressure_gradient', '—')} Pa/m",
                f"{params.get('velocity', '—')} m/s",
                f"{params.get('h', '—')} m",
                f"{params.get('R', '—')} m",
            ],
        }
        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, width='stretch', hide_index=True)

    # 3. Assumptions
    with st.expander("3️⃣  Assumptions", expanded=True):
        assumptions = params.get("assumptions") or result.assumptions
        if assumptions:
            st.markdown(
                _assumption_badges(assumptions),
                unsafe_allow_html=True,
            )
        else:
            st.info("No assumptions listed.")

    # 4. Governing Equation
    with st.expander("4️⃣  Governing Equation", expanded=True):
        st.markdown("**General Incompressible Navier–Stokes:**")
        st.latex(GENERAL_NS_LATEX)
        st.markdown("**Continuity (incompressibility):**")
        st.latex(CONTINUITY_LATEX)
        if result.governing_eq_latex:
            st.markdown("**Simplified governing equation for this problem:**")
            st.latex(result.governing_eq_latex)

    # 5. Equation Simplification
    with st.expander("5️⃣  Equation Simplification Steps", expanded=False):
        steps_key = params.get("problem_type", "poiseuille")
        steps = SIMPLIFICATION_STEPS.get(steps_key, [])
        if steps:
            for label, eq in steps:
                st.markdown(f'<div class="eq-step"><b>{label}</b></div>', unsafe_allow_html=True)
                st.latex(eq)
        else:
            st.info("Simplification steps not available for this problem type.")

    # 6. Mathematical Derivation (SymPy)
    with st.expander("6️⃣  Mathematical Derivation (SymPy)", expanded=False):
        from solver.equations import (
            solve_poiseuille_sympy, solve_couette_sympy, solve_pipe_flow_sympy
        )
        ptype = params.get("problem_type")
        mu = params.get("viscosity") or 0.001
        dpdx = params.get("pressure_gradient") or 0.0
        h = params.get("h")
        R = params.get("R")
        U = params.get("velocity") or 0.0

        try:
            if ptype == "poiseuille" and h:
                sym = solve_poiseuille_sympy(h, mu, dpdx)
                st.markdown("**General solution (symbolic):**")
                st.latex(r"u(y) = " + sym["u_latex_symbolic"])
                st.markdown("**Particular solution (with given values):**")
                st.latex(r"u(y) = " + sym["u_latex_numeric"])
            elif ptype == "couette" and h:
                sym = solve_couette_sympy(h, U, mu, dpdx)
                st.markdown("**General solution (symbolic):**")
                st.latex(r"u(y) = " + sym["u_latex_symbolic"])
                st.markdown("**Particular solution (with given values):**")
                st.latex(r"u(y) = " + sym["u_latex_numeric"])
            elif ptype == "pipe" and R:
                sym = solve_pipe_flow_sympy(R, mu, dpdx)
                st.markdown("**Analytical solution (symbolic):**")
                st.latex(r"u_z(r) = " + sym["u_latex_symbolic"])
                st.markdown("**Particular solution (with given values):**")
                st.latex(r"u_z(r) = " + sym["u_latex_numeric"])
            else:
                st.info("Symbolic derivation not available for this problem type.")
        except Exception as sym_e:
            st.warning(f"SymPy derivation error: {sym_e}")

    # 7. Numerical Calculation
    with st.expander("7️⃣  Numerical Calculation (Python/NumPy)", expanded=True):
        if result.success:
            st.markdown("**Solution formula:**")
            st.latex(result.solution_latex)
            st.markdown("**Computed results:**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    _metric_html("Maximum Velocity", _fmt(result.u_max, 5), "m/s"),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    _metric_html("Average Velocity", _fmt(result.u_avg, 5), "m/s"),
                    unsafe_allow_html=True,
                )
            with c3:
                if result.reynolds_number is not None and physical_validity is not None:
                    regime = physical_validity.flow_regime
                    regime_class = {
                        "Laminar": "regime-badge-laminar",
                        "Transitional": "regime-badge-transitional",
                        "Turbulent": "regime-badge-turbulent",
                    }.get(regime, "regime-badge-laminar")
                    st.markdown(
                        _metric_html(
                            f"Reynolds Number",
                            f"{result.reynolds_number:.2f}",
                            f'<span class="{regime_class}">{regime}</span>'
                        ),
                        unsafe_allow_html=True,
                    )
                elif result.reynolds_number is not None:
                    regime = ("Laminar" if result.reynolds_number < 2300
                              else ("Transitional" if result.reynolds_number < 4000 else "Turbulent"))
                    st.markdown(
                        _metric_html("Reynolds Number", f"{result.reynolds_number:.2f}", regime),
                        unsafe_allow_html=True,
                    )

            # Show Re formula definition explicitly
            if result.re_formula_description:
                st.markdown(
                    f'<div class="ns-card" style="margin-top:10px; font-size:0.88rem; color:#8899cc;">'
                    f'<b>Re definition:</b> {result.re_formula_description}<br>'
                    f'<b>Characteristic length:</b> {result.re_char_length_name}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            c4, c5 = st.columns(2)
            with c4:
                st.markdown(
                    _metric_html("Flow Rate", _fmt(result.flow_rate, 5, sci=True),
                                 "m³/s" if params.get("R") else "m²/s (per unit width)"),
                    unsafe_allow_html=True,
                )
            with c5:
                st.markdown(
                    _metric_html("Wall Shear Stress", _fmt(result.wall_shear_stress, 5), "Pa"),
                    unsafe_allow_html=True,
                )
        else:
            st.error(f"❌ Solver Error: {result.error_message}")

    # 8. Final Answer
    with st.expander("8️⃣  Final Answer", expanded=True):
        if result.success:
            st.markdown(
                f'<div class="ns-card ns-card-accent">'
                f'<b>Velocity Profile:</b><br>',
                unsafe_allow_html=True,
            )
            st.latex(result.solution_latex)
            rows = []
            if result.u_max is not None:
                rows.append(f"- **Maximum velocity:** {result.u_max:.6g} m/s")
            if result.u_avg is not None:
                rows.append(f"- **Average velocity:** {result.u_avg:.6g} m/s")
            if result.flow_rate is not None:
                rows.append(f"- **Volumetric flow rate:** {result.flow_rate:.4e} m³/s")
            if result.reynolds_number is not None:
                rows.append(f"- **Reynolds number:** {result.reynolds_number:.4g}")
            st.markdown("\n".join(rows))
            st.markdown("</div>", unsafe_allow_html=True)

    # 9. Verification & Physical Validity (TWO SEPARATE SECTIONS)
    with st.expander("9️⃣  Mathematical Verification + Physical Validity", expanded=True):

        # ── 9A. MATHEMATICAL VERIFICATION ──────────────────────────────────────
        st.markdown("#### 🔢 Mathematical Verification")
        st.markdown(
            "_Checks that SymPy (symbolic ODE) and NumPy (direct formula) agree at the "
            "same spatial point. Agreement means the algebra is internally consistent — "
            "it says **nothing** about whether the physical flow is actually laminar._"
        )

        if verification:
            cls = "verify-pass" if verification.passed else "verify-fail"
            icon = "✅" if verification.passed else "❌"
            st.markdown(
                f'<div class="{cls}">'
                f'<b>{icon} Mathematical check: {"PASSED" if verification.passed else "FAILED"}</b><br>'
                f'{verification.message}<br>'
                f'<small>Relative error: {verification.relative_error:.2e} '
                f'(tolerance: 1×10⁻⁶)</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")
            vc1, vc2 = st.columns(2)
            with vc1:
                st.markdown(
                    _metric_html(verification.method_a, _fmt(verification.value_a, 6), "m/s"),
                    unsafe_allow_html=True,
                )
            with vc2:
                st.markdown(
                    _metric_html(verification.method_b, _fmt(verification.value_b, 6), "m/s"),
                    unsafe_allow_html=True,
                )
        elif result.success:
            st.info("Mathematical verification not available for this problem type.")
        else:
            st.warning("Mathematical verification skipped — solver did not produce a result.")

        st.markdown("---")

        # ── 9B. PHYSICAL VALIDITY ───────────────────────────────────────────────
        st.markdown("#### 🧪 Physical Validity")
        st.markdown(
            "_Checks whether the computed Reynolds number is consistent with the "
            "**laminar-flow assumption** that underlies the analytical solution. "
            "This is independent of the mathematical check above._"
        )

        if physical_validity is not None:
            regime = physical_validity.flow_regime
            validity = physical_validity.validity_label

            # Regime badge row
            regime_class = {
                "Laminar": "regime-badge-laminar",
                "Transitional": "regime-badge-transitional",
                "Turbulent": "regime-badge-turbulent",
            }.get(regime, "regime-badge-laminar")

            pv1, pv2 = st.columns(2)
            with pv1:
                st.markdown(
                    f'<div style="padding:8px 0">'
                    f'<b>Flow regime:</b> '
                    f'<span class="{regime_class}">{regime}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with pv2:
                model_valid_icon = "✅" if regime == "Laminar" else ("⚠️" if regime == "Transitional" else "❌")
                st.markdown(
                    f'<div style="padding:8px 0">'
                    f'<b>Model validity:</b> {model_valid_icon} <b>{validity}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Re formula box
            st.markdown(
                f'<div class="ns-card" style="font-size:0.88rem; color:#8899cc; margin:8px 0;">'
                f'<b>Re = {physical_validity.re_value:.4g}</b> '
                f'&nbsp;&mdash;&nbsp; {physical_validity.re_formula_description}<br>'
                f'Characteristic length: <b>{physical_validity.char_length_name}</b><br>'
                f'Laminar limit for this flow type: Re &lt; {physical_validity.re_limit_laminar:.0f}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Validity detail box
            if regime == "Laminar":
                st.markdown(
                    f'<div class="validity-valid">'
                    f'✅ <b>Laminar assumption is physically consistent.</b><br>'
                    f'{physical_validity.detail_message}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif regime == "Transitional":
                st.markdown(
                    f'<div class="validity-warning">'
                    f'⚠️ <b>Transitional regime — model validity uncertain.</b><br><br>'
                    f'{physical_validity.detail_message}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:  # Turbulent
                st.markdown(
                    f'<div class="validity-invalid">'
                    f'❌ <b>WARNING: Laminar assumption is NOT physically valid at these parameters.</b><br><br>'
                    f'{physical_validity.detail_message}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        elif result.success:
            st.info("Physical validity check not applicable for this problem type.")
        else:
            st.warning("Physical validity check skipped — solver did not produce a result.")

    # 10. Explanation
    with st.expander("🔟  LLM Explanation", expanded=True):
        if explanation:
            st.markdown(explanation)
        else:
            st.info("No explanation available.")

    # Visualization
    st.markdown("---")
    st.markdown("### 📈 Velocity Profile Visualization")
    with st.spinner("Generating plot…"):
        fig = generate_plot(result)
    if fig:
        st.pyplot(fig)
        st.markdown(
            "<small style='color:#5566aa'>📌 Graph generated from deterministic Python calculation — not by the LLM.</small>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No visualization available for this problem type.")

    # Validity notice — updated to distinguish math from physics
    st.markdown(
        '<div class="limitation-box">'
        "<b>📋 Solution Validity Summary:</b><br>"
        "<b>Mathematical validity:</b> SymPy and NumPy agreement confirms the algebra is correct for the assumed model.<br>"
        "<b>Physical validity:</b> The flow-regime check (above) determines whether the laminar assumption is appropriate "
        "for the given parameters. A high Reynolds number means the analytical result is the <i>mathematical solution "
        "of the assumed laminar model</i>, not a description of the physically realised flow.<br><br>"
        "<b>⚠️ Limitation:</b> This system solves selected Navier–Stokes special cases. "
        "It does <b>not</b> solve the general 3D Navier–Stokes existence and smoothness problem. "
        "For turbulent flows or complex geometries, use OpenFOAM, ANSYS Fluent, or similar CFD software."
        "</div>",
        unsafe_allow_html=True,
    )


elif solve_clicked and not problem_text.strip():
    st.warning("⚠️ Please enter a problem description before clicking Solve.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#3a4060; font-size:0.82rem; padding:12px 0;'>
    🌊 LLM-Powered Navier–Stokes Solver &nbsp;|&nbsp;
    Built with Streamlit · NumPy · SymPy · OpenAI API &nbsp;|&nbsp;
    For educational and research purposes only
    </div>
    """,
    unsafe_allow_html=True,
)
