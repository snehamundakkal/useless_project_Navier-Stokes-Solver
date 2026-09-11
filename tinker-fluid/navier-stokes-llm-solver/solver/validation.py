"""
solver/validation.py
====================
Two distinct validation concerns — kept strictly separate:

  1. MATHEMATICAL VERIFICATION
     SymPy (symbolic ODE solve) vs NumPy (direct formula) at the same point.
     Agreement means the algebra is internally consistent.
     It says NOTHING about whether the assumed flow regime is physically valid.

  2. PHYSICAL VALIDITY CHECK
     Compares the deterministically computed Reynolds number against the
     laminar-flow threshold for each problem type.
     Produces a regime label and model-validity warning independently of
     the mathematical check.

These two checks are presented separately in the UI so users always know:
  - Did the math check out?   ← VerificationResult
  - Is the physics valid?     ← PhysicalValidityResult
"""

import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import Optional

from solver.equations import (
    solve_poiseuille_sympy,
    solve_couette_sympy,
    solve_pipe_flow_sympy,
)
from solver.analytical import (
    SolutionResult,
    classify_regime,
    RE_LAMINAR_LIMIT,
    RE_TRANSITIONAL_LIMIT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Mathematical Verification dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """
    MATHEMATICAL VERIFICATION ONLY.
    Checks that SymPy symbolic solution and NumPy formula agree numerically.
    Agreement here means the math is internally consistent — it says NOTHING
    about whether the physical flow is actually laminar.
    """
    passed: bool
    method_a: str   # "SymPy symbolic ODE integration"
    method_b: str   # "NumPy direct formula"
    value_a: float
    value_b: float
    relative_error: float
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Physical Validity dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysicalValidityResult:
    """
    PHYSICAL VALIDITY CHECK — completely separate from mathematical verification.

    Determines whether the computed Reynolds number is consistent with the
    laminar-flow assumption used in the analytical solution.

    A solution can be MATHEMATICALLY CORRECT (SymPy == NumPy) while being
    PHYSICALLY INVALID (Re >> laminar limit → turbulence would occur in reality).
    """
    flow_regime: str          # "Laminar" | "Transitional" | "Turbulent"
    validity_label: str       # "VALID" | "WARNING" | "OUTSIDE MODEL"
    is_physically_valid: bool # True only if regime == "Laminar"
    re_value: float
    re_limit_laminar: float
    re_limit_transitional: float
    char_length_name: str
    re_formula_description: str
    warning_message: str      # short one-liner for the UI badge
    detail_message: str       # full explanatory paragraph


# ─────────────────────────────────────────────────────────────────────────────
# Physical validity check (deterministic — LLM never involved)
# ─────────────────────────────────────────────────────────────────────────────

# Per-flow-type warning messages
_REGIME_WARNINGS = {
    ("poiseuille", "Transitional"): (
        "TRANSITIONAL REGIME: The computed Re falls in the transitional range "
        "(2300–4000 for channel flow). The laminar Poiseuille profile is mathematically "
        "derived but the flow may be intermittently turbulent in practice."
    ),
    ("poiseuille", "Turbulent"): (
        "WARNING: The requested parameters produce a high Reynolds number (Re = {Re:.1f}), "
        "computed as Re = ρ·u_avg·{L_name}/μ = {formula}. "
        "The analytical Plane Poiseuille profile is mathematically derived under fully-developed "
        "laminar assumptions, but these parameters are NOT consistent with a laminar-flow regime. "
        "The result should be interpreted as the solution of the assumed laminar model, "
        "not necessarily as a physically realized flow. "
        "Real flow at this Re would be turbulent and would require a turbulence model."
    ),
    ("couette", "Transitional"): (
        "TRANSITIONAL REGIME: Re = {Re:.1f} (Re = ρ·U·{L_name}/μ = {formula}) lies in the "
        "transitional range for Couette flow. The linear velocity profile assumes laminar conditions."
    ),
    ("couette", "Turbulent"): (
        "WARNING: Re = {Re:.1f} (Re = ρ·U·{L_name}/μ = {formula}) exceeds the practical laminar "
        "limit for Couette flow (~1000). The linear velocity profile u(y) = Uy/h is the exact "
        "solution of the assumed laminar model — but at this Re, the actual flow would be turbulent "
        "and the velocity profile would differ significantly from the linear prediction."
    ),
    ("pipe", "Transitional"): (
        "TRANSITIONAL REGIME: Re = {Re:.1f} (Re = ρ·u_avg·{L_name}/μ = {formula}) lies in the "
        "transitional range. Hagen-Poiseuille assumes laminar flow (Re < 2300)."
    ),
    ("pipe", "Turbulent"): (
        "WARNING: Re = {Re:.1f} (Re = ρ·u_avg·{L_name}/μ = {formula}) greatly exceeds the "
        "laminar limit of 2300 for pipe flow. The Hagen-Poiseuille parabolic profile is the "
        "mathematically exact solution of the assumed laminar model — but the real flow at this "
        "Re would be turbulent, with a much flatter velocity profile. "
        "Results should not be used for engineering calculations without turbulence modelling."
    ),
}

_VALIDITY_LABELS = {
    "VALID": "VALID — Laminar assumption is physically consistent",
    "WARNING": "WARNING — Transitional regime; laminar assumption may not hold",
    "OUTSIDE MODEL": "OUTSIDE MODEL — Turbulent regime; laminar solution is not physically valid",
}


def check_physical_validity(result: SolutionResult) -> Optional[PhysicalValidityResult]:
    """
    Check whether the computed Re is consistent with the laminar-flow assumption.
    Returns None if the problem type does not have a laminar assumption to check.
    All values come from the deterministic solver — LLM is not involved.
    """
    if not result.success or result.reynolds_number is None:
        return None

    ptype = result.problem_type
    if ptype not in RE_LAMINAR_LIMIT:
        return None

    Re = result.reynolds_number
    regime, validity = classify_regime(Re, ptype)
    lam_limit = RE_LAMINAR_LIMIT[ptype]
    trans_limit = RE_TRANSITIONAL_LIMIT[ptype]

    char_name = result.re_char_length_name or "L"
    formula = result.re_formula_description or f"Re = {Re:.4g}"

    # Build warning message
    if regime == "Laminar":
        warn_short = f"Re = {Re:.1f} — within laminar range (< {lam_limit:.0f})"
        detail = (
            f"The computed Reynolds number Re = {Re:.2f} "
            f"({result.re_formula_description}) "
            f"is below the laminar limit of {lam_limit:.0f} for {ptype} flow. "
            f"The laminar analytical solution is physically appropriate for these parameters."
        )
    else:
        key = (ptype, regime)
        template = _REGIME_WARNINGS.get(key, "Re = {Re:.1f} exceeds laminar limit.")
        detail = template.format(Re=Re, L_name=char_name, formula=formula)
        warn_short = (
            f"Re = {Re:.1f} — {regime.upper()} "
            f"(laminar limit: Re < {lam_limit:.0f} for {ptype} flow)"
        )

    return PhysicalValidityResult(
        flow_regime=regime,
        validity_label=validity,
        is_physically_valid=(regime == "Laminar"),
        re_value=Re,
        re_limit_laminar=lam_limit,
        re_limit_transitional=trans_limit,
        char_length_name=char_name,
        re_formula_description=formula,
        warning_message=warn_short,
        detail_message=detail,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mathematical verification helpers
# ─────────────────────────────────────────────────────────────────────────────

TOLERANCE = 1e-6   # relative tolerance for mathematical agreement


def _rel_error(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-15)
    return abs(a - b) / denom


def _sympy_eval(u_expr: sp.Expr, point: float) -> float:
    """
    Evaluate a SymPy expression at a numerical point.
    Extracts the free symbol automatically to avoid identity issues.
    """
    free = list(u_expr.free_symbols)
    if not free:
        return float(u_expr)
    sym = free[0]
    val = u_expr.subs(sym, point)
    return float(val)


def verify_poiseuille(result: SolutionResult) -> VerificationResult:
    """
    Mathematical check: SymPy symbolic ODE solve vs NumPy formula at y = h/2.
    Both evaluated at EXACTLY the same point. This is purely algebraic correctness.
    """
    h = result.params["h"]
    mu = result.params["mu"]
    dpdx = result.params["dpdx"]
    y_test = h / 2.0

    sym_data = solve_poiseuille_sympy(h, mu, dpdx)
    u_sympy = _sympy_eval(sym_data["u_numeric"], y_test)
    u_numpy = float((1.0 / (2.0 * mu)) * (-dpdx) * y_test * (h - y_test))

    rel_err = _rel_error(u_sympy, u_numpy)
    passed = rel_err < TOLERANCE

    return VerificationResult(
        passed=passed,
        method_a="SymPy (symbolic ODE integration)",
        method_b="NumPy (direct formula at y = h/2)",
        value_a=u_sympy,
        value_b=u_numpy,
        relative_error=rel_err,
        message=(
            f"u(y=h/2): SymPy = {u_sympy:.6e} m/s, NumPy = {u_numpy:.6e} m/s"
            + (" — MATCH" if passed else " — MISMATCH")
        ),
    )


def verify_couette(result: SolutionResult) -> VerificationResult:
    """Mathematical check: SymPy vs NumPy at y = h/2."""
    h = result.params["h"]
    U = result.params["U"]
    mu = result.params.get("mu", 1e-3)
    dpdx = result.params.get("dpdx", 0.0)
    y_test = h / 2.0

    sym_data = solve_couette_sympy(h, U, mu, dpdx)
    u_sympy = _sympy_eval(sym_data["u_numeric"], y_test)
    u_numpy = float((U / h) * y_test + (1.0 / (2.0 * mu)) * (-dpdx) * y_test * (h - y_test))

    rel_err = _rel_error(u_sympy, u_numpy)
    passed = rel_err < TOLERANCE

    return VerificationResult(
        passed=passed,
        method_a="SymPy (symbolic ODE integration)",
        method_b="NumPy (direct formula at y = h/2)",
        value_a=u_sympy,
        value_b=u_numpy,
        relative_error=rel_err,
        message=(
            f"u(y=h/2): SymPy = {u_sympy:.6e} m/s, NumPy = {u_numpy:.6e} m/s"
            + (" — MATCH" if passed else " — MISMATCH")
        ),
    )


def verify_pipe_flow(result: SolutionResult) -> VerificationResult:
    """Mathematical check: SymPy vs NumPy at r = R/2."""
    R = result.params["R"]
    mu = result.params["mu"]
    dpdx = result.params["dpdx"]
    r_test = R / 2.0

    sym_data = solve_pipe_flow_sympy(R, mu, dpdx)
    u_sympy = _sympy_eval(sym_data["u_numeric"], r_test)
    u_numpy = float((1.0 / (4.0 * mu)) * (-dpdx) * (R**2 - r_test**2))

    rel_err = _rel_error(u_sympy, u_numpy)
    passed = rel_err < TOLERANCE

    return VerificationResult(
        passed=passed,
        method_a="SymPy (analytical expression)",
        method_b="NumPy (direct formula at r = R/2)",
        value_a=u_sympy,
        value_b=u_numpy,
        relative_error=rel_err,
        message=(
            f"u(r=R/2): SymPy = {u_sympy:.6e} m/s, NumPy = {u_numpy:.6e} m/s"
            + (" — MATCH" if passed else " — MISMATCH")
        ),
    )


def run_verification(result: SolutionResult) -> Optional[VerificationResult]:
    """Dispatch mathematical verification based on problem type."""
    if not result.success:
        return None
    ptype = result.problem_type
    try:
        if ptype == "poiseuille":
            return verify_poiseuille(result)
        elif ptype == "couette":
            return verify_couette(result)
        elif ptype == "pipe":
            return verify_pipe_flow(result)
        else:
            return None
    except Exception as exc:
        return VerificationResult(
            passed=False,
            method_a="SymPy",
            method_b="NumPy",
            value_a=float("nan"),
            value_b=float("nan"),
            relative_error=float("nan"),
            message=f"Verification error: {exc}",
        )
