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

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Background ── */
    .stApp { background: linear-gradient(135deg, #0a0d1a 0%, #0f1529 50%, #0a1020 100%); }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1022 0%, #111628 100%);
        border-right: 1px solid #1e2340;
    }

    /* ── Cards ── */
    .ns-card {
        background: linear-gradient(135deg, #12172a 0%, #1a2040 100%);
        border: 1px solid #1e2d60;
        border-radius: 16px;
        padding: 24px 28px;
        margin: 12px 0;
        box-shadow: 0 4px 24px rgba(79,142,247,0.08);
    }
    .ns-card-accent {
        border-left: 4px solid #4f8ef7;
    }

    /* ── Hero title ── */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4f8ef7, #a78bfa, #f7c94f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #8892b0;
        margin-bottom: 2rem;
        font-style: italic;
    }

    /* ── Example cards ── */
    .example-card {
        background: linear-gradient(135deg, #13182e 0%, #1c2444 100%);
        border: 1px solid #1e2d60;
        border-radius: 12px;
        padding: 16px;
        margin: 6px 0;
        cursor: pointer;
        transition: all 0.25s ease;
    }
    .example-card:hover {
        border-color: #4f8ef7;
        box-shadow: 0 0 20px rgba(79,142,247,0.20);
    }
    .example-title { font-size: 0.95rem; font-weight: 600; color: #ccd6f6; }
    .example-desc  { font-size: 0.80rem; color: #7888aa; margin-top: 4px; }

    /* ── Assumption badges ── */
    .assumption-badge {
        display: inline-block;
        background: rgba(61,220,132,0.12);
        border: 1px solid rgba(61,220,132,0.3);
        color: #3ddc84;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 3px 4px;
        font-size: 0.82rem;
    }

    /* ── Flag badges ── */
    .flag-badge {
        display: inline-block;
        background: rgba(247,196,79,0.12);
        border: 1px solid rgba(247,196,79,0.3);
        color: #f7c94f;
        border-radius: 8px;
        padding: 4px 10px;
        margin: 3px 0;
        font-size: 0.82rem;
    }

    /* ── Result metric boxes ── */
    .metric-box {
        background: linear-gradient(135deg, #151c35 0%, #1d2648 100%);
        border: 1px solid #2a3560;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        margin: 6px 2px;
    }
    .metric-label { font-size: 0.78rem; color: #7888aa; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #4f8ef7; font-family: 'JetBrains Mono', monospace; }
    .metric-unit  { font-size: 0.78rem; color: #5566aa; }

    /* ── Verification ── */
    .verify-pass {
        background: rgba(61,220,132,0.10);
        border: 1px solid rgba(61,220,132,0.4);
        border-radius: 10px;
        padding: 14px;
        color: #3ddc84;
    }
    .verify-fail {
        background: rgba(247,84,84,0.10);
        border: 1px solid rgba(247,84,84,0.4);
        border-radius: 10px;
        padding: 14px;
        color: #f75454;
    }

    /* ── Demo mode banner ── */
    .demo-banner {
        background: linear-gradient(90deg, rgba(247,196,79,0.15), rgba(247,196,79,0.05));
        border: 1px solid rgba(247,196,79,0.4);
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 16px;
        color: #f7c94f;
        font-size: 0.9rem;
    }

    /* ── Physical validity boxes ── */
    .validity-valid {
        background: rgba(61,220,132,0.10);
        border: 1px solid rgba(61,220,132,0.4);
        border-radius: 10px;
        padding: 14px 18px;
        color: #3ddc84;
        font-size: 0.92rem;
    }
    .validity-warning {
        background: rgba(247,196,79,0.12);
        border: 1px solid rgba(247,196,79,0.5);
        border-left: 4px solid #f7c94f;
        border-radius: 10px;
        padding: 14px 18px;
        color: #f7c94f;
        font-size: 0.92rem;
    }
    .validity-invalid {
        background: rgba(247,84,84,0.12);
        border: 1px solid rgba(247,84,84,0.5);
        border-left: 4px solid #f75454;
        border-radius: 10px;
        padding: 14px 18px;
        color: #f75454;
        font-size: 0.92rem;
    }
    /* ── Regime badge ── */
    .regime-badge-laminar    { background: rgba(61,220,132,0.18); border: 1px solid #3ddc84; color: #3ddc84; border-radius: 20px; padding: 3px 14px; font-size: 0.82rem; font-weight: 600; }
    .regime-badge-transitional { background: rgba(247,196,79,0.18); border: 1px solid #f7c94f; color: #f7c94f; border-radius: 20px; padding: 3px 14px; font-size: 0.82rem; font-weight: 600; }
    .regime-badge-turbulent  { background: rgba(247,84,84,0.18); border: 1px solid #f75454; color: #f75454; border-radius: 20px; padding: 3px 14px; font-size: 0.82rem; font-weight: 600; }

    /* ── Limitation notice ── */
    .limitation-box {
        background: rgba(167,139,250,0.08);
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 10px;
        padding: 14px;
        color: #a78bfa;
        font-size: 0.88rem;
        margin-top: 16px;
    }

    /* ── Solve button ── */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #3b6ff5, #5a3fcc);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        width: 100%;
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #5a86ff, #7a5fff);
        box-shadow: 0 6px 24px rgba(79,142,247,0.40);
        transform: translateY(-1px);
    }

    /* Expander headers */
    .streamlit-expanderHeader {
        background: #12172a !important;
        border-radius: 8px !important;
    }

    /* ── Divider ── */
    hr { border-color: #1e2340; }

    /* ── Scrollable equation area ── */
    .eq-scroll { overflow-x: auto; max-width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    '<div class="hero-title">🌊 LLM Navier–Stokes Solver</div>'
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
