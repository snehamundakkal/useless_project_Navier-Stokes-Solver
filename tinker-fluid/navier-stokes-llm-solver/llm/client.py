"""
llm/client.py
=============
LLM API abstraction layer.
Supports OpenAI API. Falls back to Demo Mode if no API key is configured.
Works on both local (.env) and Streamlit Cloud (st.secrets).
"""

import os
import json
import time
from typing import Tuple, Optional

from dotenv import load_dotenv
from llm.prompts import (
    SYSTEM_PROMPT,
    build_extraction_prompt,
    build_explanation_prompt,
    build_demo_explanation,
)
from llm.parser import parse_llm_response, extract_parameters_for_solver

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """
    Read a secret from st.secrets (Streamlit Cloud) first,
    then fall back to os.environ / .env (local development).
    """
    try:
        import streamlit as st
        val = st.secrets.get(key, None)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

LLM_PROVIDER = _get_secret("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY", "")
OPENAI_MODEL = _get_secret("OPENAI_MODEL", "gpt-4o-mini")
DEMO_MODE_ENV = _get_secret("DEMO_MODE", "false").lower() == "true"


def is_demo_mode() -> bool:
    """Return True if the app should run in Demo Mode (no LLM API calls)."""
    if _get_secret("DEMO_MODE", "false").lower() == "true":
        return True
    api_key = _get_secret("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Client
# ─────────────────────────────────────────────────────────────────────────────

def _call_openai(messages: list, temperature: float = 0.1,
                 model: Optional[str] = None, max_retries: int = 2) -> Tuple[str, Optional[str]]:
    """
    Call the OpenAI Chat Completions API.

    Returns:
        (response_text, error_message)
    """
    try:
        from openai import OpenAI, APIError, APITimeoutError, AuthenticationError
    except ImportError:
        return "", "openai package not installed. Run: pip install openai"

    client = OpenAI(api_key=_get_secret("OPENAI_API_KEY", ""))
    used_model = model or _get_secret("OPENAI_MODEL", "gpt-4o-mini")

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=used_model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content or "", None

        except AuthenticationError:
            return "", "Invalid OpenAI API key. Please check your .env file."
        except APITimeoutError:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return "", "OpenAI API request timed out. Please try again."
        except APIError as e:
            return "", f"OpenAI API error: {e}"
        except Exception as e:
            return "", f"Unexpected error calling OpenAI: {e}"

    return "", "Max retries exceeded."


# ─────────────────────────────────────────────────────────────────────────────
# Demo Mode Responses
# ─────────────────────────────────────────────────────────────────────────────

# Pre-built structured JSON responses for Demo Mode
DEMO_RESPONSES = {
    "poiseuille": {
        "problem_type": "poiseuille",
        "geometry": "Two stationary parallel plates with fluid between them",
        "fluid": "water",
        "density": 1000.0,
        "density_unit": "kg/m3",
        "viscosity": 0.001,
        "viscosity_unit": "Pa.s",
        "velocity": None,
        "velocity_unit": "m/s",
        "pressure_gradient": -100.0,
        "pressure_gradient_unit": "Pa/m",
        "dimensions": {"h": 0.02, "R": None, "L": None},
        "boundary_conditions": ["u = 0 at y = 0 (lower plate)", "u = 0 at y = h (upper plate)"],
        "initial_conditions": [],
        "assumptions": [
            "Steady flow (∂/∂t = 0)",
            "Incompressible fluid (ρ = const)",
            "Newtonian fluid",
            "Fully developed flow (∂u/∂x = 0)",
            "1D flow (v = 0)",
            "No-slip boundary condition",
            "Negligible body forces",
        ],
        "governing_equation": r"\mu \frac{d^2 u}{d y^2} = \frac{dp}{dx}",
        "solution_method": "Double integration of the simplified N-S ODE with no-slip BCs",
        "required_calculations": ["velocity profile u(y)", "maximum velocity", "average velocity", "flow rate"],
        "flags": [],
        "problem_summary": "Pressure-driven laminar flow of water between two stationary parallel plates.",
        "supported": True,
    },
    "couette": {
        "problem_type": "couette",
        "geometry": "Two parallel plates, upper plate moving at velocity U",
        "fluid": "oil",
        "density": 900.0,
        "density_unit": "kg/m3",
        "viscosity": 0.01,
        "viscosity_unit": "Pa.s",
        "velocity": 0.5,
        "velocity_unit": "m/s",
        "pressure_gradient": 0.0,
        "pressure_gradient_unit": "Pa/m",
        "dimensions": {"h": 0.01, "R": None, "L": None},
        "boundary_conditions": ["u = 0 at y = 0 (lower plate)", "u = U at y = h (upper plate)"],
        "initial_conditions": [],
        "assumptions": [
            "Steady flow",
            "Incompressible fluid",
            "Newtonian fluid",
            "Fully developed flow",
            "Zero pressure gradient",
            "No-slip at both plates",
        ],
        "governing_equation": r"\mu \frac{d^2 u}{d y^2} = 0",
        "solution_method": "Double integration of d²u/dy²=0 with moving-plate BCs",
        "required_calculations": ["velocity profile u(y)", "shear stress", "Reynolds number"],
        "flags": [],
        "problem_summary": "Couette flow between two parallel plates with the upper plate moving at 0.5 m/s.",
        "supported": True,
    },
    "pipe": {
        "problem_type": "pipe",
        "geometry": "Circular pipe, fully developed laminar flow",
        "fluid": "water",
        "density": 1000.0,
        "density_unit": "kg/m3",
        "viscosity": 0.001,
        "viscosity_unit": "Pa.s",
        "velocity": None,
        "velocity_unit": "m/s",
        "pressure_gradient": -200.0,
        "pressure_gradient_unit": "Pa/m",
        "dimensions": {"h": None, "R": 0.025, "L": None},
        "boundary_conditions": ["u_z = 0 at r = R (no-slip)", "du_z/dr = 0 at r = 0 (symmetry)"],
        "initial_conditions": [],
        "assumptions": [
            "Steady, axisymmetric flow",
            "Incompressible fluid",
            "Newtonian fluid",
            "Fully developed flow",
            "Laminar (Re < 2300)",
            "No-slip at pipe wall",
            "Negligible body forces",
        ],
        "governing_equation": r"\frac{1}{r}\frac{d}{dr}\left(r\frac{du_z}{dr}\right) = \frac{1}{\mu}\frac{dp}{dz}",
        "solution_method": "Hagen-Poiseuille analytical solution in cylindrical coordinates",
        "required_calculations": ["velocity profile u(r)", "maximum velocity", "volumetric flow rate", "Reynolds number"],
        "flags": [],
        "problem_summary": "Hagen-Poiseuille laminar flow in a circular pipe driven by a pressure gradient.",
        "supported": True,
    },
    "reynolds": {
        "problem_type": "reynolds",
        "geometry": "Circular pipe",
        "fluid": "water",
        "density": 1000.0,
        "density_unit": "kg/m3",
        "viscosity": 0.001,
        "viscosity_unit": "Pa.s",
        "velocity": 1.0,
        "velocity_unit": "m/s",
        "pressure_gradient": None,
        "pressure_gradient_unit": "Pa/m",
        "dimensions": {"h": None, "R": 0.025, "L": None},
        "boundary_conditions": [],
        "initial_conditions": [],
        "assumptions": [
            "Characteristic velocity = mean pipe velocity",
            "Characteristic length = pipe diameter D = 2R",
        ],
        "governing_equation": r"Re = \frac{\rho V D}{\mu}",
        "solution_method": "Direct computation of Re = ρVD/μ",
        "required_calculations": ["Reynolds number", "flow regime"],
        "flags": [],
        "problem_summary": "Calculate the Reynolds number for water flowing in a circular pipe.",
        "supported": True,
    },
}


def get_demo_parsed_response(problem_hint: str) -> dict:
    """
    Return a demo structured response based on keyword matching in the problem hint.
    """
    hint = problem_hint.lower()
    if "reynolds" in hint or "re =" in hint or "regime" in hint:
        return DEMO_RESPONSES["reynolds"].copy()
    elif "couette" in hint or "moving" in hint or "upper plate" in hint or "moves" in hint:
        return DEMO_RESPONSES["couette"].copy()
    elif "pipe" in hint or "cylinder" in hint or "circular" in hint or "radius" in hint:
        return DEMO_RESPONSES["pipe"].copy()
    else:
        return DEMO_RESPONSES["poiseuille"].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_problem_parameters(
    user_problem: str,
    temperature: float = 0.05,
    model: Optional[str] = None,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Step 1: Send the problem to the LLM for understanding and parameter extraction.

    Returns:
        (solver_params_dict, error_message)
    """
    if is_demo_mode():
        demo = get_demo_parsed_response(user_problem)
        params = extract_parameters_for_solver(demo)
        return params, None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_extraction_prompt(user_problem)},
    ]

    raw_text, err = _call_openai(messages, temperature=temperature, model=model)
    if err:
        return None, err

    parsed, parse_err = parse_llm_response(raw_text)
    if parse_err:
        return None, f"LLM response parsing failed: {parse_err}"

    params = extract_parameters_for_solver(parsed)
    return params, None


def generate_explanation(
    user_problem: str,
    params: dict,
    numerical_results: dict,
    re_info: dict = None,
    temperature: float = 0.3,
    model: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Step 2: Ask the LLM to explain the deterministic solver results in plain language.
    re_info carries the deterministic Re + regime so the LLM cannot invent the flow regime.

    Returns:
        (explanation_text, error_message)
    """
    problem_type = params.get("problem_type", "unknown")

    if is_demo_mode():
        explanation = build_demo_explanation(problem_type, params, numerical_results)
        return explanation, None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_explanation_prompt(
                user_problem, params, numerical_results, re_info=re_info
            ),
        },
    ]

    text, err = _call_openai(messages, temperature=temperature, model=model)
    if err:
        fallback = build_demo_explanation(problem_type, params, numerical_results)
        return fallback, err

    return text, None
