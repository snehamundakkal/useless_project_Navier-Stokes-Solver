"""
solver/analytical.py
====================
Analytical (closed-form) solutions for supported Navier-Stokes special cases.
All calculations are performed in Python/NumPy — NOT by the LLM.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SolutionResult:
    """Container for a solved flow problem."""
    problem_type: str
    success: bool
    error_message: str = ""

    # Physical parameters
    params: dict = field(default_factory=dict)

    # Velocity profile arrays
    y_points: Optional[np.ndarray] = None   # spatial coordinate
    u_profile: Optional[np.ndarray] = None  # velocity profile

    # Scalar results
    u_max: Optional[float] = None
    u_avg: Optional[float] = None
    flow_rate: Optional[float] = None
    reynolds_number: Optional[float] = None
    wall_shear_stress: Optional[float] = None

    # Reynolds number metadata — set by solver, used for physical validity check
    re_char_velocity: Optional[float] = None    # velocity used in Re formula
    re_char_length: Optional[float] = None      # length used in Re formula
    re_char_length_name: str = ""               # e.g. "h (plate separation)" or "D=2R (pipe diameter)"
    re_formula_description: str = ""            # human-readable description of Re definition

    # Equation info
    governing_eq_latex: str = ""
    solution_latex: str = ""
    assumptions: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Laminar validity thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Re_critical for each flow type (below = laminar, above = suspect/invalid)
# References: White (2015), Batchelor (2000)
RE_LAMINAR_LIMIT = {
    # Plane channel (Re = ρ·u_avg·h/μ): linear stability onset ~7696 (Orr-Sommerfeld),
    # but in practice transition occurs at Re ~ 1000-3000 depending on disturbances.
    # We use 2300 as a practical conservative limit (consistent with pipe flow literature).
    "poiseuille": 2300.0,
    # Couette (Re = ρ·U·h/μ): linear stability onset ~infinity (subcritical transition),
    # but in practice turbulence appears above Re ~ 360-1000. Use 1000 as conservative limit.
    "couette": 1000.0,
    # Hagen-Poiseuille (Re = ρ·u_avg·D/μ): classical laminar limit
    "pipe": 2300.0,
}

RE_TRANSITIONAL_LIMIT = {
    "poiseuille": 4000.0,
    "couette": 2000.0,
    "pipe": 4000.0,
}


def classify_regime(Re: float, problem_type: str) -> tuple[str, str]:
    """
    Return (regime_label, model_validity_label) for a given Re and problem type.

    Returns:
        regime_label   : "Laminar" | "Transitional" | "Turbulent"
        validity_label : "VALID" | "WARNING" | "OUTSIDE MODEL"
    """
    lam_limit = RE_LAMINAR_LIMIT.get(problem_type, 2300.0)
    trans_limit = RE_TRANSITIONAL_LIMIT.get(problem_type, 4000.0)

    if Re <= lam_limit:
        return "Laminar", "VALID"
    elif Re <= trans_limit:
        return "Transitional", "WARNING"
    else:
        return "Turbulent", "OUTSIDE MODEL"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Plane Poiseuille Flow  (pressure-driven, both plates stationary)
# ─────────────────────────────────────────────────────────────────────────────

def solve_plane_poiseuille(h: float, mu: float, dpdx: float,
                            rho: float = 1000.0, n_points: int = 100) -> SolutionResult:
    """
    Plane Poiseuille flow between two stationary parallel plates.

    Parameters
    ----------
    h     : plate separation [m]
    mu    : dynamic viscosity [Pa·s]
    dpdx  : pressure gradient dp/dx [Pa/m]  (negative for flow in +x)
    rho   : density [kg/m³]

    Governing equation (after simplification):
        μ d²u/dy² = dp/dx

    Solution:
        u(y) = (1/2μ)(-dp/dx) · y(h - y)

    Reynolds number definition:
        Re = ρ · u_avg · h / μ
        where h is the full plate separation (characteristic length).
        Laminar assumption requires Re < 2300 (practical limit).
    """
    if h <= 0 or mu <= 0:
        return SolutionResult("poiseuille", False,
                               "Plate separation h and viscosity μ must be positive.")

    y = np.linspace(0, h, n_points)
    u = (1.0 / (2.0 * mu)) * (-dpdx) * y * (h - y)

    u_max = float((-dpdx) * h**2 / (8.0 * mu))
    u_avg = float((-dpdx) * h**2 / (12.0 * mu))
    Q_per_unit = u_avg * h          # [m²/s] per unit width
    tau_w = float(abs(dpdx) * h / 2.0)   # wall shear stress
    # Re based on u_avg and h (full plate separation as characteristic length)
    Re = float(rho * u_avg * h / mu) if u_avg > 0 else 0.0

    return SolutionResult(
        problem_type="poiseuille",
        success=True,
        params={"h": h, "mu": mu, "dpdx": dpdx, "rho": rho},
        y_points=y,
        u_profile=u,
        u_max=u_max,
        u_avg=u_avg,
        flow_rate=Q_per_unit,
        reynolds_number=Re,
        wall_shear_stress=tau_w,
        re_char_velocity=u_avg,
        re_char_length=h,
        re_char_length_name="h (full plate separation)",
        re_formula_description=f"Re = ρ·u_avg·h/μ = {rho:.4g}×{u_avg:.4g}×{h:.4g}/{mu:.4g}",
        governing_eq_latex=r"\mu \frac{d^2 u}{d y^2} = \frac{dp}{dx}",
        solution_latex=r"u(y) = \frac{1}{2\mu}\left(-\frac{dp}{dx}\right) y(h - y)",
        assumptions=[
            "Steady flow (∂/∂t = 0)",
            "Incompressible fluid (ρ = const)",
            "Newtonian fluid (τ = μ du/dy)",
            "Fully developed flow (∂u/∂x = 0)",
            "1D flow (v = w = 0)",
            "No-slip boundary condition (u = 0 at both plates)",
            "Negligible body forces",
            "ASSUMED laminar (requires Re < 2300 for physical validity)",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Couette Flow  (upper plate moves, zero pressure gradient)
# ─────────────────────────────────────────────────────────────────────────────

def solve_couette(h: float, U: float, mu: float = 1e-3,
                  rho: float = 1000.0, dpdx: float = 0.0,
                  n_points: int = 100) -> SolutionResult:
    """
    Couette flow: upper plate moves at velocity U, lower plate stationary.
    Optional pressure gradient dpdx can be added (generalised Couette flow).

    Pure Couette (dpdx=0):  u(y) = U*y/h
    General:  u(y) = U*y/h + (1/2μ)(-dp/dx) * y*(h-y)

    Reynolds number definition:
        Re = ρ · U · h / μ
        Laminar assumption requires Re < 1000 (practical limit for Couette).
    """
    if h <= 0 or U < 0:
        return SolutionResult("couette", False,
                               "h must be positive; U must be non-negative.")

    y = np.linspace(0, h, n_points)
    # Linear Couette term + parabolic pressure-driven term
    u = (U / h) * y + (1.0 / (2.0 * mu)) * (-dpdx) * y * (h - y)

    u_max = float(np.max(u))
    u_avg = float(np.mean(u))
    Q_per_unit = u_avg * h
    tau_w = float(mu * U / h + abs(dpdx) * h / 2.0)
    # Re based on plate velocity U and gap h (characteristic velocity and length)
    Re = float(rho * U * h / mu) if U > 0 else 0.0

    if dpdx == 0.0:
        sol_latex = r"u(y) = \frac{U}{h} y"
        gov_eq_latex = r"\mu \frac{d^2 u}{d y^2} = 0"
    else:
        sol_latex = r"u(y) = \frac{U}{h} y + \frac{1}{2\mu}\left(-\frac{dp}{dx}\right) y(h - y)"
        gov_eq_latex = r"\mu \frac{d^2 u}{d y^2} = \frac{dp}{dx}"

    return SolutionResult(
        problem_type="couette",
        success=True,
        params={"h": h, "U": U, "mu": mu, "dpdx": dpdx, "rho": rho},
        y_points=y,
        u_profile=u,
        u_max=u_max,
        u_avg=u_avg,
        flow_rate=Q_per_unit,
        reynolds_number=Re,
        wall_shear_stress=tau_w,
        re_char_velocity=U,
        re_char_length=h,
        re_char_length_name="h (plate gap)",
        re_formula_description=f"Re = ρ·U·h/μ = {rho:.4g}×{U:.4g}×{h:.4g}/{mu:.4g}",
        governing_eq_latex=gov_eq_latex,
        solution_latex=sol_latex,
        assumptions=[
            "Steady flow (∂/∂t = 0)",
            "Incompressible fluid (ρ = const)",
            "Newtonian fluid (τ = μ du/dy)",
            "Fully developed flow (∂u/∂x = 0)",
            "1D flow (v = 0)",
            "No-slip BC: u(0) = 0, u(h) = U",
            "Negligible body forces" if dpdx == 0 else "Applied pressure gradient dp/dx",
            "ASSUMED laminar (requires Re < 1000 for physical validity)",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hagen-Poiseuille (pipe flow)
# ─────────────────────────────────────────────────────────────────────────────

def solve_pipe_flow(R: float, mu: float, dpdx: float,
                    rho: float = 1000.0, n_points: int = 100) -> SolutionResult:
    """
    Fully developed laminar flow in a circular pipe (Hagen-Poiseuille).

    u(r) = (1/4μ)(-dp/dx)(R² - r²)

    Reynolds number definition:
        Re = ρ · u_avg · D / μ   where D = 2R (pipe diameter)
        Laminar for Re < 2300 (classical criterion).
    """
    if R <= 0 or mu <= 0:
        return SolutionResult("pipe", False,
                               "Pipe radius R and viscosity μ must be positive.")

    r = np.linspace(0, R, n_points)
    u = (1.0 / (4.0 * mu)) * (-dpdx) * (R**2 - r**2)

    u_max = float((-dpdx) * R**2 / (4.0 * mu))
    u_avg = float((-dpdx) * R**2 / (8.0 * mu))   # = u_max / 2
    Q = float(np.pi * (-dpdx) * R**4 / (8.0 * mu))
    D = 2.0 * R
    # Re based on u_avg and D (pipe diameter — the standard hydraulic diameter definition)
    Re = float(rho * u_avg * D / mu) if u_avg > 0 else 0.0
    tau_w = float(abs(dpdx) * R / 2.0)

    return SolutionResult(
        problem_type="pipe",
        success=True,
        params={"R": R, "mu": mu, "dpdx": dpdx, "rho": rho},
        y_points=r,        # radial coordinate
        u_profile=u,
        u_max=u_max,
        u_avg=u_avg,
        flow_rate=Q,
        reynolds_number=Re,
        wall_shear_stress=tau_w,
        re_char_velocity=u_avg,
        re_char_length=D,
        re_char_length_name="D = 2R (pipe diameter)",
        re_formula_description=f"Re = ρ·u_avg·D/μ = {rho:.4g}×{u_avg:.4g}×{D:.4g}/{mu:.4g}",
        governing_eq_latex=(
            r"\frac{1}{r}\frac{d}{dr}\left(r\frac{du_z}{dr}\right) = \frac{1}{\mu}\frac{dp}{dz}"
        ),
        solution_latex=r"u_z(r) = \frac{1}{4\mu}\left(-\frac{dp}{dz}\right)(R^2 - r^2)",
        assumptions=[
            "Steady flow (∂/∂t = 0)",
            "Incompressible fluid (ρ = const)",
            "Newtonian fluid",
            "Fully developed flow (∂u_z/∂z = 0)",
            "Axisymmetric flow",
            "No-slip at pipe wall (u_z(R) = 0)",
            "ASSUMED laminar (requires Re < 2300 for physical validity)",
            "Negligible body forces",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reynolds Number Calculator
# ─────────────────────────────────────────────────────────────────────────────

def compute_reynolds(rho: float, velocity: float, length: float, mu: float,
                     length_name: str = "L") -> SolutionResult:
    """
    Compute Reynolds number: Re = ρ V L / μ

    Parameters
    ----------
    rho      : density [kg/m³]
    velocity : characteristic velocity [m/s]
    length   : characteristic length [m]
    mu       : dynamic viscosity [Pa·s]
    length_name : label for the characteristic length (e.g. "D=2R")
    """
    if mu <= 0 or length <= 0 or rho <= 0:
        return SolutionResult("reynolds", False,
                               "Density, length, and viscosity must be positive.")

    Re = float(rho * velocity * length / mu)
    regime = "Laminar" if Re < 2300 else ("Transitional" if Re < 4000 else "Turbulent")

    return SolutionResult(
        problem_type="reynolds",
        success=True,
        params={"rho": rho, "V": velocity, "L": length, "mu": mu},
        reynolds_number=Re,
        re_char_velocity=velocity,
        re_char_length=length,
        re_char_length_name=length_name,
        re_formula_description=f"Re = ρ·V·{length_name}/μ = {rho:.4g}×{velocity:.4g}×{length:.4g}/{mu:.4g}",
        governing_eq_latex=r"Re = \frac{\rho V L}{\mu} = \frac{V L}{\nu}",
        solution_latex=rf"Re = \frac{{{rho:.4g} \times {velocity:.4g} \times {length:.4g}}}{{{mu:.4g}}} = {Re:.4g}",
        assumptions=[
            f"Characteristic velocity V = {velocity} m/s",
            f"Characteristic length {length_name} = {length} m",
            f"Flow regime (deterministic): {regime}",
        ],
    )
