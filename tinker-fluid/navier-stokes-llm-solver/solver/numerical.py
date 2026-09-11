"""
solver/numerical.py
===================
NumPy / SciPy numerical helpers for derived quantities.
All calculations are deterministic — not performed by the LLM.
"""

import numpy as np
from typing import Optional


def pressure_drop(dpdx: float, L: float) -> float:
    """ΔP = |dp/dx| * L"""
    return abs(dpdx) * L


def kinematic_viscosity(mu: float, rho: float) -> float:
    """ν = μ / ρ  [m²/s]"""
    return mu / rho


def wall_shear_poiseuille(dpdx: float, h: float) -> float:
    """τ_w = |dp/dx| * h/2  for Poiseuille flow"""
    return abs(dpdx) * h / 2.0


def wall_shear_pipe(dpdx: float, R: float) -> float:
    """τ_w = |dp/dx| * R/2  for pipe flow"""
    return abs(dpdx) * R / 2.0


def darcy_weisbach_friction(Re: float) -> Optional[float]:
    """
    Darcy-Weisbach friction factor for laminar pipe flow.
    f = 64 / Re
    """
    if Re <= 0:
        return None
    return 64.0 / Re


def entry_length_laminar(D: float, Re: float) -> float:
    """
    Hydrodynamic entry length for laminar pipe flow.
    L_e = 0.06 * Re * D
    """
    return 0.06 * Re * D


def poiseuille_velocity_at(y: float, h: float, mu: float, dpdx: float) -> float:
    """u(y) for Poiseuille flow at a specific y."""
    return (1.0 / (2.0 * mu)) * (-dpdx) * y * (h - y)


def couette_velocity_at(y: float, h: float, U: float, mu: float = 1e-3,
                         dpdx: float = 0.0) -> float:
    """u(y) for Couette (+ optional pressure-driven) flow at a specific y."""
    return (U / h) * y + (1.0 / (2.0 * mu)) * (-dpdx) * y * (h - y)


def pipe_velocity_at(r: float, R: float, mu: float, dpdx: float) -> float:
    """u(r) for Hagen-Poiseuille pipe flow at radius r."""
    return (1.0 / (4.0 * mu)) * (-dpdx) * (R**2 - r**2)


def compute_flow_rate_plate(u_profile: np.ndarray, y_points: np.ndarray) -> float:
    """Volumetric flow rate per unit width via numerical integration (trapezoidal)."""
    return float(np.trapz(u_profile, y_points))


def compute_flow_rate_pipe(u_profile: np.ndarray, r_points: np.ndarray) -> float:
    """Volumetric flow rate for pipe: Q = 2π ∫ u(r) r dr"""
    return float(2.0 * np.pi * np.trapz(u_profile * r_points, r_points))
