"""
visualization/plots.py
======================
Matplotlib-based velocity profile plots.
All data comes from the deterministic Python solver — NOT from the LLM.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from typing import Optional

from solver.analytical import SolutionResult


# ─────────────────────────────────────────────────────────────────────────────
# Shared style
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG   = "#0f1117"
CARD_BG   = "#1a1d27"
ACCENT    = "#4f8ef7"
ACCENT2   = "#f7c94f"
TEXT      = "#e8eaf6"
GRID      = "#2a2d3a"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "grid.color": GRID,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


def _styled_fig(width: float = 7, height: float = 5) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.grid(True, alpha=0.4)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 1. Plate-flow profile  (Poiseuille & Couette)
# ─────────────────────────────────────────────────────────────────────────────

def plot_plate_flow(result: SolutionResult) -> Figure:
    """
    Plot the velocity profile u(y) for plate-based flows.
    The profile is displayed horizontally (u on x-axis, y on y-axis)
    to give a physically intuitive orientation.
    """
    y = result.y_points
    u = result.u_profile
    h = result.params.get("h", y[-1])

    fig, ax = _styled_fig()

    # Fill the flow region
    ax.fill_betweenx(y, 0, u, alpha=0.18, color=ACCENT, label="Velocity distribution")

    # Main profile line
    ax.plot(u, y, color=ACCENT, linewidth=2.5, label="u(y)")

    # Mark maximum velocity
    if result.u_max is not None:
        idx_max = np.argmax(u)
        ax.scatter(u[idx_max], y[idx_max], s=80, color=ACCENT2, zorder=5,
                   label=f"u_max = {result.u_max:.4f} m/s")
        ax.axvline(result.u_max, color=ACCENT2, linestyle=":", alpha=0.6)

    # Plate lines
    ax.axhline(0, color="#aaaaaa", linewidth=2, label="Lower plate (y = 0)")
    ax.axhline(h, color="#cccccc", linewidth=2, label=f"Upper plate (y = {h} m)")

    # Labels & decoration
    ax.set_xlabel("Velocity u(y)  [m/s]", fontsize=11)
    ax.set_ylabel("Distance y  [m]", fontsize=11)
    ptype = result.problem_type.capitalize()
    ax.set_title(f"{ptype} Flow — Velocity Profile", fontsize=13)
    ax.legend(loc="upper left", fontsize=9,
              framealpha=0.2, edgecolor=GRID)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pipe-flow profile  (Hagen-Poiseuille)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pipe_flow(result: SolutionResult) -> Figure:
    """
    Plot the parabolic velocity profile for Hagen-Poiseuille pipe flow.
    Shows both the r-side profile and a 2D cross-sectional color map.
    """
    r = result.y_points   # radial coordinate (0 to R)
    u = result.u_profile
    R = result.params.get("R", r[-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax in (ax1, ax2):
        ax.set_facecolor(CARD_BG)
        ax.grid(True, alpha=0.4)

    # — Left panel: u(r) profile —
    r_full = np.concatenate([-r[::-1], r])
    u_full = np.concatenate([u[::-1], u])
    ax1.fill_betweenx(r_full, 0, u_full, alpha=0.18, color=ACCENT)
    ax1.plot(u_full, r_full, color=ACCENT, linewidth=2.5, label="u(r)")
    if result.u_max is not None:
        ax1.scatter(result.u_max, 0, s=80, color=ACCENT2, zorder=5,
                    label=f"u_max = {result.u_max:.4f} m/s")
    ax1.axvline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Velocity u(r)  [m/s]", fontsize=11)
    ax1.set_ylabel("Radial position r  [m]", fontsize=11)
    ax1.set_title("Velocity Profile u(r)", fontsize=12)
    ax1.legend(fontsize=9, framealpha=0.2, edgecolor=GRID)

    # — Right panel: 2D cross-section colormap —
    theta = np.linspace(0, 2 * np.pi, 200)
    r_2d = np.linspace(0, R, 100)
    T, Rv = np.meshgrid(theta, r_2d)
    X = Rv * np.cos(T)
    Y = Rv * np.sin(T)
    U_2d = (1.0 / (4.0 * result.params["mu"])) * (-result.params["dpdx"]) * (R**2 - Rv**2)
    cm = ax2.contourf(X, Y, U_2d, levels=30, cmap="plasma")
    cbar = fig.colorbar(cm, ax=ax2)
    cbar.set_label("u  [m/s]", color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x  [m]", fontsize=11)
    ax2.set_ylabel("y  [m]", fontsize=11)
    ax2.set_title("Cross-Section Velocity Distribution", fontsize=12)

    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Hagen-Poiseuille Pipe Flow", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reynolds number regime gauge
# ─────────────────────────────────────────────────────────────────────────────

def plot_reynolds_gauge(Re: float) -> Figure:
    """
    Horizontal bar gauge showing flow regime for the computed Reynolds number.
    """
    fig, ax = _styled_fig(width=8, height=2.5)

    # Regime regions
    ax.barh(0, 2300, color="#3ddc84", alpha=0.8, height=0.6, label="Laminar (Re < 2300)")
    ax.barh(0, 1700, left=2300, color="#f7c94f", alpha=0.8, height=0.6,
            label="Transitional (2300–4000)")
    ax.barh(0, 16000, left=4000, color="#f75454", alpha=0.8, height=0.6,
            label="Turbulent (Re > 4000)")

    # Marker for computed Re
    re_plot = min(Re, 19500)
    ax.axvline(re_plot, color="white", linewidth=3, zorder=5)
    ax.scatter(re_plot, 0, s=120, color="white", zorder=6)

    label_y = 0.38
    ax.annotate(f"Re = {Re:.0f}", xy=(re_plot, 0), xytext=(re_plot, label_y),
                ha="center", fontsize=11, fontweight="bold", color="white",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5))

    ax.set_xlim(0, 20000)
    ax.set_ylim(-0.5, 0.8)
    ax.set_xlabel("Reynolds Number", fontsize=11)
    ax.set_yticks([])
    ax.set_title("Flow Regime Classification", fontsize=13)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.2, edgecolor=GRID)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def generate_plot(result: SolutionResult) -> Optional[Figure]:
    """Return the appropriate Matplotlib Figure for a given SolutionResult."""
    if not result.success:
        return None
    ptype = result.problem_type
    try:
        if ptype in ("poiseuille", "couette"):
            return plot_plate_flow(result)
        elif ptype == "pipe":
            return plot_pipe_flow(result)
        elif ptype == "reynolds" and result.reynolds_number is not None:
            return plot_reynolds_gauge(result.reynolds_number)
        return None
    except Exception:
        return None
