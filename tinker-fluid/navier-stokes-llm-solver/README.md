# 🌊 LLM-Powered Navier–Stokes Problem Solver

> **Ask a fluid mechanics problem. Let the LLM interpret it. Let mathematics verify it.**

---

## Project Motivation

The Navier–Stokes equations govern all viscous fluid motion. While the **existence and smoothness** of solutions to the general 3D Navier–Stokes system remains an unsolved Millennium Prize problem, many **special cases** admit exact analytical solutions under well-stated assumptions.

This project combines a Large Language Model (LLM) with a deterministic Python/SymPy mathematical engine to:
1. **Interpret** natural-language fluid mechanics problems
2. **Extract** physical parameters and identify assumptions
3. **Solve** the appropriate simplified Navier–Stokes equations analytically
4. **Verify** results independently using both SymPy (symbolic) and NumPy (numerical) methods
5. **Explain** the physics in accessible language

**⚠️ Important:** This system does NOT solve the general 3D Navier–Stokes problem. It solves specific, well-defined special cases under stated assumptions.

---

## What Are the Navier–Stokes Equations?

For an incompressible Newtonian fluid, the Navier–Stokes equations are:

```
∂u/∂t + (u · ∇)u = -(1/ρ)∇p + ν∇²u + f

∇ · u = 0   (continuity / incompressibility)
```

Where:
- **u** = velocity vector field [m/s]
- **ρ** = density [kg/m³]
- **p** = pressure [Pa]
- **ν = μ/ρ** = kinematic viscosity [m²/s]
- **f** = body force per unit mass [m/s²]

---

## System Architecture

```
User Input (natural language)
        ↓
    LLM MODEL
    ├── Problem understanding
    ├── Parameter extraction → structured JSON
    ├── Assumption identification
    └── Equation/form selection
        ↓
   PYTHON ENGINE (deterministic)
    ├── SymPy   → symbolic differential equation solving
    ├── NumPy   → numerical evaluation of velocity profiles
    └── SciPy   → additional numerical methods if needed
        ↓
   VERIFICATION MODULE
    ├── SymPy result vs. NumPy result
    └── Relative error check (tolerance 1×10⁻⁶)
        ↓
    LLM MODEL
    └── Explanation of mathematical results
        ↓
   STREAMLIT UI
    ├── Given parameters table
    ├── Assumptions display
    ├── Governing equation (LaTeX)
    ├── Simplification steps
    ├── SymPy derivation
    ├── Numerical results
    ├── Verification status
    ├── LLM explanation
    └── Matplotlib velocity profile
```

---

## LLM Role

The LLM serves two distinct purposes:

1. **Parameter Extraction** (Step 1): Interprets the natural-language problem, extracts physical parameters (viscosity, density, geometry, boundary conditions), classifies the problem type, and returns **structured JSON**.

2. **Explanation** (Step 2): After the Python engine computes results, the LLM explains the mathematical results in physical terms — what the velocity profile means, why the parabolic shape arises, engineering significance, etc.

The LLM **never** performs arithmetic, integration, or differentiation. Those operations are handled deterministically by NumPy/SymPy.

---

## Python Solver Role

The Python solver (`solver/`) performs **all numerical and symbolic computations**:

| Module | Role |
|--------|------|
| `equations.py` | SymPy symbolic solving of ODEs with boundary conditions |
| `analytical.py` | NumPy-based analytical solutions for all supported flow types |
| `numerical.py` | Helper functions for derived quantities (Re, τ_w, Q, etc.) |
| `validation.py` | Independent cross-verification: SymPy vs. NumPy |

---

## SymPy Role

SymPy performs **exact symbolic mathematics**:
- Solves differential equations (e.g., `μ d²u/dy² = dp/dx`)
- Applies boundary conditions algebraically
- Returns exact analytical expressions
- These are then compared with NumPy numerical evaluations

---

## Verification System

The verification module compares two independent calculations:
1. **SymPy** evaluates the symbolic solution at a test point (e.g., y = h/2)
2. **NumPy** reads the corresponding value from the numerical profile array

If these agree within tolerance 1×10⁻⁶ (relative), the solution is marked **VERIFIED ✓**.

This demonstrates that the LLM's parameter extraction was physically consistent — not that the LLM computed the answer.

---

## Installation

```bash
# Clone or download the project
cd navier-stokes-llm-solver

# Install dependencies
pip install -r requirements.txt
```

---

## API Configuration

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and set your API key:
OPENAI_API_KEY=sk-...your-key...
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
```

**If you don't have an API key:** The app runs in Demo Mode automatically.

---

## Demo Mode

Demo Mode provides a **fully functional demonstration** without an LLM API key:
- Pre-built parameter extraction responses for all 4 example problems
- Full Python solver pipeline runs normally
- Verification runs normally
- Pre-written educational explanations are displayed
- All visualizations are generated

To activate: leave `OPENAI_API_KEY` blank in `.env`, or toggle "Force Demo Mode" in the sidebar.

---

## Usage

```bash
# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Example Problems

| # | Problem | Type |
|---|---------|------|
| 1 | Water flows between two stationary plates (h=0.02 m, dp/dx=-100 Pa/m, μ=0.001 Pa·s) | Poiseuille |
| 2 | Upper plate moves at 0.5 m/s, oil between plates (h=0.01 m, μ=0.01 Pa·s) | Couette |
| 3 | Water in circular pipe (R=0.025 m, dp/dz=-200 Pa/m, μ=0.001 Pa·s) | Pipe Flow |
| 4 | Reynolds number for water pipe flow (V=1.0 m/s, D=0.05 m, μ=0.001 Pa·s) | Reynolds |

---

## Supported Problem Types

| Type | Description | Governing Equation |
|------|-------------|-------------------|
| Poiseuille | Pressure-driven, stationary plates | μ d²u/dy² = dp/dx |
| Couette | Moving plate, viscous drag | d²u/dy² = 0 |
| Pipe Flow | Hagen-Poiseuille | (1/r) d/dr(r du/dr) = dp/dz / μ |
| Reynolds | Flow regime classification | Re = ρVL/μ |

---

## Limitations

1. **Not a general CFD solver** — only solves selected analytical special cases
2. **Laminar flow only** — all solutions assume Re is below transition
3. **Newtonian fluids only** — no non-Newtonian rheology
4. **Fully developed flow** — inlet/outlet effects are not modelled
5. **LLM can make errors** — parameter extraction is verified by schema validation
6. **2D flows only** (for plate configurations)
7. **Does NOT solve** the 3D Navier–Stokes existence and smoothness problem

---

## References

1. White, F. M. (2015). *Fluid Mechanics* (8th ed.). McGraw-Hill.
2. Batchelor, G. K. (2000). *An Introduction to Fluid Dynamics*. Cambridge University Press.
3. Munson, B. R., Young, D. F., & Okiishi, T. H. (2012). *Fundamentals of Fluid Mechanics*. Wiley.
4. SymPy Development Team (2024). *SymPy: Python library for symbolic mathematics*. https://sympy.org
5. Harris, C. R. et al. (2020). *Array programming with NumPy*. Nature, 585, 357–362.
6. Streamlit Inc. (2024). *Streamlit: A faster way to build and share data apps*. https://streamlit.io
7. OpenAI (2024). *OpenAI API Documentation*. https://platform.openai.com/docs
8. Stokes, G. G. (1845). On the theories of the internal friction of fluids in motion. *Trans. Cambridge Phil. Soc.*, 8, 287–319.
9. Hagen, G. (1839). Über die Bewegung des Wassers in engen cylindrischen Röhren. *Ann. Phys.*, 122, 423–442.
