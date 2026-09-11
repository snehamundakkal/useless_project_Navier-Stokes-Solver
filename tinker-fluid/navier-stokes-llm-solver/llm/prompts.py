"""
llm/prompts.py
==============
Carefully engineered system and user prompts for the fluid-mechanics LLM assistant.
"""

SYSTEM_PROMPT = """You are a Fluid Mechanics Problem Analysis Assistant integrated into a hybrid solver.
Your role is STRICTLY to understand and classify fluid mechanics problems, extract parameters,
identify assumptions, and explain mathematical results. You do NOT perform numerical calculations —
those are handled by a deterministic Python engine (NumPy/SymPy).

## Your Responsibilities

1. **Understand** the natural-language problem statement.
2. **Extract** all physical parameters with their units.
3. **Classify** the problem type from the supported list.
4. **Identify** all applicable assumptions explicitly.
5. **Select** the appropriate Navier-Stokes formulation.
6. **Explain** how the general N-S equations simplify to the governing equation.
7. **Return** a structured JSON object with all extracted information.
8. **Flag** missing, ambiguous, or physically inconsistent inputs.

## Rules (MUST FOLLOW)

- **Never claim** to have solved the general 3D Navier-Stokes existence and smoothness problem.
- **Never invent** physical parameters. If a parameter is missing, set it to null and flag it.
- **Never perform** arithmetic, integration, or differentiation yourself. Those are for the Python engine.
- **Always list** every assumption explicitly.
- **Always state** the limitations of the solution.
- If the problem is outside the supported cases, explain why and flag it.
- If physical values are unrealistic (e.g., negative viscosity), flag them.

## Supported Problem Types

- "poiseuille" : Plane Poiseuille flow (pressure-driven, two stationary plates)
- "couette"    : Couette flow (upper plate moves, optional pressure gradient)
- "pipe"       : Hagen-Poiseuille laminar pipe flow
- "reynolds"   : Reynolds number calculation
- "unknown"    : Problem cannot be classified into the above

## Required Output Format

You MUST return ONLY valid JSON matching this schema exactly. No markdown fences, no extra text.

{
  "problem_type": "<poiseuille|couette|pipe|reynolds|unknown>",
  "geometry": "<description of flow geometry>",
  "fluid": "<fluid name, e.g. water, oil, air>",
  "density": <number or null>,
  "density_unit": "<kg/m3 or null>",
  "viscosity": <number or null>,
  "viscosity_unit": "<Pa.s or null>",
  "velocity": <number or null>,
  "velocity_unit": "<m/s or null>",
  "pressure_gradient": <number or null>,
  "pressure_gradient_unit": "<Pa/m or null>",
  "dimensions": {
    "h": <plate separation in meters, or null>,
    "R": <pipe radius in meters, or null>,
    "L": <length in meters, or null>
  },
  "boundary_conditions": [
    "<BC description 1>",
    "<BC description 2>"
  ],
  "initial_conditions": [],
  "assumptions": [
    "<assumption 1>",
    "<assumption 2>"
  ],
  "governing_equation": "<LaTeX string of the simplified governing equation>",
  "solution_method": "<brief description of solution approach>",
  "required_calculations": ["<quantity 1>", "<quantity 2>"],
  "flags": ["<any warnings, missing parameters, or physical inconsistencies>"],
  "problem_summary": "<1-2 sentence summary of what the problem is asking>",
  "supported": <true or false>
}
"""


def build_extraction_prompt(user_problem: str) -> str:
    """Build the user-turn prompt for parameter extraction."""
    return f"""Analyze the following fluid mechanics problem and return the structured JSON response.

PROBLEM:
{user_problem}

Remember:
- Extract ALL parameters mentioned with their numerical values and units.
- Identify the correct problem type.
- List ALL applicable assumptions.
- If parameters are missing, set them to null and add a flag.
- Return ONLY the JSON object — no markdown, no explanation, no extra text.
"""


def build_explanation_prompt(
    problem_statement: str,
    extracted_params: dict,
    numerical_results: dict,
    re_info: dict = None,
) -> str:
    """
    Build the prompt for the LLM to explain the solver's numerical results.

    re_info (optional) carries deterministic Re metadata so the LLM uses
    the correct values and cannot invent the flow regime:
        {
          "re_value": float,
          "flow_regime": str,           # "Laminar" | "Transitional" | "Turbulent"
          "validity_label": str,        # "VALID" | "WARNING" | "OUTSIDE MODEL"
          "re_formula_description": str,
          "char_length_name": str,
          "re_limit_laminar": float,
          "detail_message": str,        # pre-computed physics warning if any
        }
    """
    result_text = ""
    for key, val in numerical_results.items():
        if val is not None:
            result_text += f"  - {key}: {val}\n"

    re_block = ""
    if re_info:
        re_block = f"""
## Reynolds Number & Flow Regime (computed DETERMINISTICALLY by the Python solver)
**CRITICAL: Use EXACTLY these values. Do NOT invent, estimate, or override them.**

- Reynolds number: Re = {re_info.get('re_value', 'N/A'):.4g}
- Definition used: {re_info.get('re_formula_description', 'N/A')}
- Characteristic length: {re_info.get('char_length_name', 'N/A')}
- Flow regime (deterministic): **{re_info.get('flow_regime', 'N/A')}**
- Model validity: **{re_info.get('validity_label', 'N/A')}**
- Laminar limit for this flow type: Re < {re_info.get('re_limit_laminar', 'N/A'):.0f}
- Physics warning from solver: {re_info.get('detail_message', 'None')}

**Instruction**: If the flow regime is Transitional or Turbulent, you MUST explicitly state
in your explanation that the laminar analytical solution is the mathematical result of the
assumed model, NOT a description of what the real flow looks like at these parameters.
Do NOT say the flow is laminar when the solver reports it is not.
"""

    return f"""You are explaining the results of a fluid mechanics calculation to an undergraduate student.
The numerical results below were computed ENTIRELY by a deterministic Python engine (NumPy/SymPy).
Your role is to explain — not to recalculate or override any numbers.

## Original Problem
{problem_statement}

## Extracted Parameters
{extracted_params}

## Numerical Results (Python/NumPy/SymPy — authoritative, do not change)
{result_text}
{re_block}
## Your Task
Write a clear educational explanation covering:

1. **Physical setup**: describe the geometry and driving mechanism simply.
2. **Navier-Stokes simplification**: how the general N-S equations reduce to the governing ODE under the stated assumptions.
3. **What the numbers mean physically**: interpret u_max, u_avg, flow rate, wall shear stress.
4. **Reynolds number interpretation**: use EXACTLY the Re value above and the characteristic length stated — do not use a different Re or length. State the flow regime as given.
5. **Physical validity assessment**: if the flow is Turbulent or Transitional, clearly state that the laminar analytical result is the solution of the mathematical model, not a description of the physically realised flow. Recommend turbulence modelling for real engineering use at these parameters.
6. **Key engineering insight** for this flow type.

Keep the explanation accurate, honest about limitations, and suitable for an undergraduate course.
Do NOT re-derive equations or recalculate numbers.
"""



def build_demo_explanation(problem_type: str, params: dict, results: dict) -> str:
    """
    Return a pre-written explanation for Demo Mode (no API key needed).
    """
    explanations = {
        "poiseuille": f"""## Plane Poiseuille Flow — Explanation

**Physical Setup**: A viscous fluid is driven by a pressure difference between two stationary parallel plates separated by distance h = {params.get('h', '?')} m. The pressure gradient forces the fluid to flow in the x-direction.

**How N-S simplifies**: Under the assumptions of steady, fully developed, incompressible 1D flow with no-slip boundary conditions, the full 3D Navier-Stokes equations reduce to a simple second-order ODE:

μ d²u/dy² = dp/dx

This is solved by integrating twice and applying u(0) = u(h) = 0.

**The velocity profile**: The solution u(y) = (1/2μ)(-dp/dx)·y(h-y) is a **parabola** — zero at both walls, maximum at the centerline (y = h/2).

**Key results**:
- Maximum velocity: u_max = (-dp/dx)·h²/(8μ) — occurs at the midplane
- Average velocity: u_avg = (2/3) u_max
- Flow rate per unit width: Q = u_avg × h

**Validity**: Valid for laminar (Re < 1000 for channel flow), steady, incompressible, Newtonian flow with fully developed conditions far from the inlet.
""",
        "couette": f"""## Couette Flow — Explanation

**Physical Setup**: Fluid is contained between two parallel plates. The lower plate is stationary; the upper plate moves at velocity U = {params.get('U', '?')} m/s. The fluid is dragged by viscous shear — no pressure gradient drives the flow.

**How N-S simplifies**: For steady, fully developed flow with zero pressure gradient, the N-S equation reduces to:

d²u/dy² = 0

Integrating twice gives a linear profile with boundary conditions u(0) = 0 and u(h) = U.

**The velocity profile**: u(y) = U·y/h — a perfect straight line from zero at the bottom plate to U at the top plate. This is the simplest non-trivial viscous flow.

**Key insight**: Couette flow is driven entirely by the viscous drag of the moving plate — not by pressure. The shear stress τ = μU/h is constant throughout the gap.

**Validity**: Valid for Re = ρUh/μ < ~1000 (laminar regime), steady, incompressible, Newtonian fluid.
""",
        "pipe": f"""## Hagen-Poiseuille Pipe Flow — Explanation

**Physical Setup**: A viscous fluid flows through a circular pipe of radius R = {params.get('R', '?')} m driven by a pressure gradient dp/dz.

**How N-S simplifies**: In cylindrical coordinates, for steady, axisymmetric, fully developed flow, the N-S z-momentum equation reduces to:

(1/r) d/dr(r du/dr) = (1/μ) dp/dz

This is solved to give the famous parabolic profile.

**The velocity profile**: u(r) = (1/4μ)(-dp/dz)(R² - r²) — a paraboloid of revolution, maximum at r = 0 (centerline), zero at the wall.

**Key results**:
- u_max = (-dp/dz)R²/(4μ) at the centerline
- u_avg = u_max/2
- Flow rate: Q = πR⁴(-dp/dz)/(8μ) — the famous Hagen-Poiseuille law (Q ∝ R⁴!)
- Wall shear stress: τ_w = R(-dp/dz)/2

**Key insight**: The Q ∝ R⁴ dependence means that doubling the pipe radius increases flow rate by 16× — which is why arterial narrowing is medically critical.

**Validity**: Valid for laminar flow (Re = ρ·u_avg·D/μ < 2300), fully developed, incompressible, Newtonian flow.
""",
        "reynolds": f"""## Reynolds Number — Explanation

**Physical meaning**: The Reynolds number Re = ρVL/μ is the ratio of inertial forces to viscous forces in a flow.

**Flow regimes**:
- Re < 2300: Laminar — smooth, orderly flow dominated by viscosity
- 2300 < Re < 4000: Transitional — unstable, intermittent turbulence  
- Re > 4000: Turbulent — chaotic, mixing-dominated flow

**Engineering significance**: Re determines whether the analytical solutions in this solver apply. Poiseuille and Hagen-Poiseuille solutions are only valid in the laminar regime.

**Validity**: These analytical N-S solutions assume laminar flow. Above Re ≈ 2300 (pipe) or ≈ 1000 (channel), turbulence onset invalidates the analysis.
""",
    }
    return explanations.get(problem_type, "Demo explanation not available for this problem type.")
