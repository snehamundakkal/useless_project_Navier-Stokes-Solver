# Limitations — LLM-Powered Navier–Stokes Solver

## ⚠️ Critical Statement

**This system does NOT solve the general 3D Navier–Stokes existence and smoothness problem.**

The existence and uniqueness of smooth solutions to the 3D Navier–Stokes equations is an unsolved Millennium Prize Problem (Clay Mathematics Institute, 2000). No software system, LLM, or combination thereof can claim to solve it.

This application solves **specific, well-defined special cases** under explicitly stated simplifying assumptions.

---

## Scope Limitations

### 1. Not a General-Purpose CFD Solver
This application does not:
- Perform discretisation (FEM, FVM, FDM)
- Handle complex geometries
- Solve turbulent flows
- Model heat transfer or compressibility
- Handle time-dependent (unsteady) problems numerically

For general CFD, use: OpenFOAM, ANSYS Fluent, SU2, or Basilisk.

### 2. Supported Cases Only
Only these flow configurations are supported:
- Plane Poiseuille flow (2D, pressure-driven, parallel plates)
- Couette flow (2D, plate-driven, parallel plates)
- Hagen-Poiseuille flow (axisymmetric, circular pipe)
- Reynolds number calculation

### 3. Laminar Flow Only
All analytical solutions assume **laminar, fully developed** flow. The solutions are physically invalid for:
- Re > 2300 (pipe flow)
- Re > ~1000 (channel flow)
- Transitional or turbulent flow

### 4. Newtonian Fluids Only
Non-Newtonian fluids (polymers, blood, slurries, Bingham plastics, power-law fluids) are not supported.

### 5. Fully Developed Flow
Solutions assume conditions far from pipe/channel inlets where the velocity profile has become unchanging in the streamwise direction. Entry-length effects are not modelled.

### 6. 2D / Axisymmetric
Plate-flow solutions are 2D (infinite-width assumption). True 3D effects (secondary flows, end walls) are ignored.

---

## LLM Limitations

### 7. Parameter Extraction Errors
The LLM may:
- Misinterpret problem statements (especially ambiguous or poorly worded ones)
- Assign incorrect units
- Miss parameters that are implied but not stated
- Classify problems incorrectly

**Mitigation:** Schema validation, flag system, and independent numerical verification.

### 8. LLM Arithmetic Errors
LLMs are known to make arithmetic mistakes. This is why:
- The LLM **never** performs calculations in this system
- All numbers come from Python/NumPy/SymPy
- Results are verified independently

### 9. Context Window
Very complex, multi-part problems may exceed context or confuse the extraction step. Keep problems focused on a single configuration.

### 10. Hallucination
The LLM may occasionally generate plausible-sounding but incorrect physics explanations. Critical numerical results should always be verified against textbook sources.

---

## Numerical Limitations

### 11. Tolerance
The verification tolerance is set to 1×10⁻⁶ (relative error). This is appropriate for the simple analytical functions involved. SymPy vs. NumPy agreement is expected to be near machine precision.

### 12. Grid Resolution
Velocity profiles use 100 points by default. For very thin gaps (h << 1 mm) or high Re, profile resolution may be coarse. This does not affect the analytical formula — only the plotted graph.

---

## What This Project IS

- A demonstration of **hybrid LLM + deterministic solver** architecture
- An educational tool for **undergraduate fluid mechanics**
- A showcase of **LLM + SymPy + NumPy + verification** pipeline design
- An honest implementation with explicit limitation disclosure

## What This Project IS NOT

- A research-grade CFD solver
- A general Navier–Stokes solver
- A turbulence modelling tool
- A replacement for ANSYS Fluent, OpenFOAM, or similar

---

## References

- Clay Mathematics Institute. (2000). *Navier–Stokes Existence and Smoothness*. Millennium Prize Problems. https://www.claymath.org/millennium-problems/
- Fefferman, C. L. (2000). *Existence and smoothness of the Navier–Stokes equation*. Clay Math. Inst. Millennium Prize Problems.
