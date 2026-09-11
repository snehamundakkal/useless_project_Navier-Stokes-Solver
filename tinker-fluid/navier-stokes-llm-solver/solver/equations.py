"""
solver/equations.py
===================
SymPy symbolic representations of the Navier-Stokes equations and their
simplified forms for common flow configurations.
"""

import sympy as sp

# ─────────────────────────────────────────────────────────────────────────────
# Common symbolic variables
# ─────────────────────────────────────────────────────────────────────────────
y, x, r, t = sp.symbols("y x r t", real=True)
u = sp.Function("u")
mu, rho, nu = sp.symbols("mu rho nu", positive=True)
dp_dx, dp_dr = sp.symbols("dp_dx dp_dr", real=True)
U, h, R = sp.symbols("U h R", positive=True)


# ─────────────────────────────────────────────────────────────────────────────
# General N-S (incompressible, displayed as LaTeX strings)
# ─────────────────────────────────────────────────────────────────────────────

GENERAL_NS_LATEX = (
    r"\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = "
    r"-\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}"
)

CONTINUITY_LATEX = r"\nabla \cdot \mathbf{u} = 0"

# ─────────────────────────────────────────────────────────────────────────────
# Plane Poiseuille / Couette (between parallel plates, flow in x-direction)
# Governing ODE after simplification:  mu * d²u/dy² = dp/dx
# ─────────────────────────────────────────────────────────────────────────────

POISEUILLE_ODE_LATEX = r"\mu \frac{d^2 u}{d y^2} = \frac{dp}{dx}"
POISEUILLE_SIMPLIFIED_LATEX = r"\frac{d^2 u}{d y^2} = \frac{1}{\mu}\frac{dp}{dx}"

def solve_poiseuille_sympy(h_val: float, mu_val: float, dpdx_val: float):
    """
    Solve Plane Poiseuille flow symbolically with SymPy.

    μ d²u/dy² = dp/dx,   u(0) = u(h) = 0

    Returns:
        u_expr   : SymPy expression for u(y)
        latex_sol: LaTeX string of the solution
        constants: dict of computed integration constants
    """
    u_fn = sp.Function("u")
    y_sym = sp.Symbol("y", real=True, nonneg=True)
    mu_sym = sp.Symbol("mu", positive=True)
    dpdx_sym = sp.Symbol("dp_dx", real=True)
    h_sym = sp.Symbol("h", positive=True)

    ode = sp.Eq(mu_sym * u_fn(y_sym).diff(y_sym, 2), dpdx_sym)
    general = sp.dsolve(ode, u_fn(y_sym))

    # Apply BCs: u(0)=0, u(h)=0
    C1, C2 = sp.symbols("C1 C2")
    expr = general.rhs
    bc0 = expr.subs(y_sym, 0)         # u(0) = 0
    bch = expr.subs(y_sym, h_sym)     # u(h) = 0
    sol_consts = sp.solve([bc0, bch - 0], [C1, C2])
    u_expr = expr.subs(sol_consts)

    # Substitute numerical values for display
    u_num = u_expr.subs([(mu_sym, mu_val), (dpdx_sym, dpdx_val), (h_sym, h_val)])
    u_num = sp.simplify(u_num)

    # Also build nice symbolic form
    u_sym_nice = sp.simplify(u_expr)

    return {
        "u_symbolic": u_sym_nice,
        "u_numeric": u_num,
        "u_latex_symbolic": sp.latex(u_sym_nice),
        "u_latex_numeric": sp.latex(u_num),
        "constants": sol_consts,
    }


def solve_couette_sympy(h_val: float, U_val: float, mu_val: float = None, dpdx_val: float = 0.0):
    """
    Solve general Couette flow (upper plate moves at U, optional pressure gradient).

    μ d²u/dy² = dp/dx,   u(0) = 0,  u(h) = U

    Pure Couette: dp/dx = 0  →  u(y) = U*y/h
    """
    y_sym = sp.Symbol("y", real=True, nonneg=True)
    mu_sym = sp.Symbol("mu", positive=True)
    dpdx_sym = sp.Symbol("dp_dx", real=True)
    h_sym = sp.Symbol("h", positive=True)
    U_sym = sp.Symbol("U", positive=True)

    u_fn = sp.Function("u")
    ode = sp.Eq(mu_sym * u_fn(y_sym).diff(y_sym, 2), dpdx_sym)
    general = sp.dsolve(ode, u_fn(y_sym))

    C1, C2 = sp.symbols("C1 C2")
    expr = general.rhs
    bc0 = expr.subs(y_sym, 0)          # u(0) = 0
    bch = expr.subs(y_sym, h_sym) - U_sym  # u(h) = U
    sol_consts = sp.solve([bc0, bch], [C1, C2])
    u_expr = sp.simplify(expr.subs(sol_consts))

    subs_dict = {mu_sym: mu_val if mu_val else 1.0,
                 dpdx_sym: dpdx_val,
                 h_sym: h_val,
                 U_sym: U_val}
    u_num = sp.simplify(u_expr.subs(subs_dict))

    return {
        "u_symbolic": u_expr,
        "u_numeric": u_num,
        "u_latex_symbolic": sp.latex(u_expr),
        "u_latex_numeric": sp.latex(u_num),
        "constants": sol_consts,
    }


def solve_pipe_flow_sympy(R_val: float, mu_val: float, dpdx_val: float):
    """
    Solve Hagen-Poiseuille flow in a circular pipe symbolically.

    Governing ODE (cylindrical coords, after simplification):
        mu * (1/r) * d/dr(r * du/dr) = dp/dx

    Simplified:   d²u/dr² + (1/r) du/dr = (1/mu) dp/dx

    BCs:  du/dr = 0 at r=0 (symmetry),  u(R) = 0 (no-slip)

    Analytical solution:  u(r) = -(1/4mu)(dp/dx)(R² - r²)
    """
    r_sym = sp.Symbol("r", real=True, nonneg=True)
    mu_sym = sp.Symbol("mu", positive=True)
    dpdx_sym = sp.Symbol("dp_dx", real=True)
    R_sym = sp.Symbol("R", positive=True)

    # Analytical form directly (SymPy Euler ODE solver can be tricky here)
    # u(r) = (1/(4*mu)) * (-dp/dx) * (R^2 - r^2)
    u_sym = (1 / (4 * mu_sym)) * (-dpdx_sym) * (R_sym**2 - r_sym**2)
    u_num = u_sym.subs([(mu_sym, mu_val), (dpdx_sym, dpdx_val), (R_sym, R_val)])
    u_num = sp.simplify(u_num)

    return {
        "u_symbolic": u_sym,
        "u_numeric": u_num,
        "u_latex_symbolic": sp.latex(u_sym),
        "u_latex_numeric": sp.latex(u_num),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX strings for simplification steps
# ─────────────────────────────────────────────────────────────────────────────

SIMPLIFICATION_STEPS = {
    "poiseuille": [
        ("General N-S (x-momentum)",
         r"\rho\left(\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y}\right) = -\frac{\partial p}{\partial x} + \mu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)"),
        ("Steady flow (∂/∂t = 0)",
         r"\rho\left(u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y}\right) = -\frac{\partial p}{\partial x} + \mu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)"),
        ("Fully developed flow (∂u/∂x = 0, v = 0)",
         r"0 = -\frac{\partial p}{\partial x} + \mu\frac{\partial^2 u}{\partial y^2}"),
        ("Rearranged",
         r"\mu \frac{d^2 u}{d y^2} = \frac{dp}{dx}"),
        ("Integrated twice",
         r"u(y) = \frac{1}{2\mu}\frac{dp}{dx}y^2 + C_1 y + C_2"),
        ("Apply BCs: u(0)=0, u(h)=0",
         r"u(y) = -\frac{1}{2\mu}\frac{dp}{dx}(hy - y^2) = \frac{1}{2\mu}\left(-\frac{dp}{dx}\right)y(h - y)"),
    ],
    "couette": [
        ("General N-S (x-momentum)", r"\rho\left(\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y}\right) = -\frac{\partial p}{\partial x} + \mu\frac{\partial^2 u}{\partial y^2}"),
        ("Steady, fully developed, no pressure gradient", r"\mu \frac{d^2 u}{d y^2} = 0"),
        ("Integrated twice", r"u(y) = C_1 y + C_2"),
        ("Apply BCs: u(0)=0, u(h)=U", r"u(y) = \frac{U}{h}y"),
    ],
    "pipe": [
        ("General N-S (r-component, cylindrical)", r"\rho\left(\frac{\partial u_z}{\partial t} + u_r\frac{\partial u_z}{\partial r}\right) = -\frac{\partial p}{\partial z} + \mu\left(\frac{\partial^2 u_z}{\partial r^2} + \frac{1}{r}\frac{\partial u_z}{\partial r}\right)"),
        ("Steady, fully developed, axisymmetric flow", r"0 = -\frac{dp}{dz} + \mu\frac{1}{r}\frac{d}{dr}\left(r\frac{du_z}{dr}\right)"),
        ("Rearranged",
         r"\frac{1}{r}\frac{d}{dr}\left(r\frac{du_z}{dr}\right) = \frac{1}{\mu}\frac{dp}{dz}"),
        ("Integrated twice (with regularity at r=0, no-slip at r=R)",
         r"u_z(r) = \frac{1}{4\mu}\left(-\frac{dp}{dz}\right)(R^2 - r^2)"),
    ],
}
