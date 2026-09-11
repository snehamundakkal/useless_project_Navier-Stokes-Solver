# Architecture — LLM-Powered Navier–Stokes Solver

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT FRONTEND                          │
│  ┌─────────────┐  ┌───────────────┐  ┌─────────────────────────┐   │
│  │ Example Cards│  │ Problem Input │  │   Results (10 sections) │   │
│  └─────────────┘  └───────────────┘  └─────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ User problem text
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          LLM LAYER  (llm/)                          │
│                                                                     │
│  client.py ──► prompts.py (system + user prompts)                  │
│      │                                                              │
│      ▼                                                              │
│  OpenAI API  (or Demo Mode fallback)                                │
│      │                                                              │
│      ▼                                                              │
│  parser.py ──► JSON validation + schema enforcement                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Validated parameter dict
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SOLVER LAYER  (solver/)                      │
│                                                                     │
│  equations.py ──► SymPy ODE solving + LaTeX generation              │
│  analytical.py ──► NumPy velocity profile computation               │
│  numerical.py ──► Derived quantities (Re, Q, τ_w, etc.)             │
│  validation.py ──► Independent cross-verification                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ SolutionResult + VerificationResult
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     VISUALIZATION  (visualization/)                 │
│                                                                     │
│  plots.py ──► Matplotlib: velocity profiles, Reynolds gauge         │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Matplotlib Figure
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM EXPLANATION  (llm/)                          │
│                                                                     │
│  client.py ──► explanation prompt with numerical results            │
│      │                                                              │
│      ▼                                                              │
│  OpenAI API  (or Demo Mode fallback)                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### `app.py`
- Streamlit UI layout and state management
- Orchestrates the full pipeline
- Renders LaTeX, tables, metrics, and plots
- **No mathematical computation**

### `llm/client.py`
- Provider abstraction (OpenAI / Demo Mode)
- API error handling and retry logic
- Calls `extract_problem_parameters()` and `generate_explanation()`

### `llm/prompts.py`
- System prompt engineering
- Extraction prompt template
- Explanation prompt template
- Demo mode pre-written responses

### `llm/parser.py`
- JSON extraction from LLM text (handles markdown fences)
- Schema validation
- Type coercion for numeric fields
- Flags for missing/invalid parameters

### `solver/equations.py`
- SymPy symbolic ODE solving
- LaTeX string generation for equations
- Simplification step definitions

### `solver/analytical.py`
- Closed-form NumPy solutions
- `SolutionResult` dataclass
- Solvers for: Poiseuille, Couette, Pipe, Reynolds

### `solver/numerical.py`
- Helper computations (Q, τ_w, f, L_entry, etc.)
- Point-evaluation functions for validation

### `solver/validation.py`
- Runs SymPy and NumPy independently
- Compares at a reference point
- Returns `VerificationResult` with pass/fail and relative error

### `visualization/plots.py`
- Dark-themed Matplotlib plots
- Plate flow: horizontal u(y) profile
- Pipe flow: u(r) + 2D cross-section colormap
- Reynolds: regime gauge bar chart

### `examples/example_problems.py`
- `ExampleProblem` dataclass
- 4 pre-built, tested problem statements
- Used by UI buttons and Demo Mode

## Data Flow

```
str (problem text)
    → params: dict (LLM JSON extraction)
    → result: SolutionResult (Python solver)
    → verification: VerificationResult (SymPy vs NumPy)
    → explanation: str (LLM)
    → fig: matplotlib.Figure (visualization)
    → Streamlit render
```

## Error Handling Strategy

| Layer | Error | Handling |
|-------|-------|----------|
| LLM | API key invalid | Switch to Demo Mode |
| LLM | Timeout | Retry up to 2×, then Demo fallback |
| LLM | Malformed JSON | `parser.py` extraction + fallback to Demo |
| Solver | Missing params | Return `SolutionResult(success=False, error_message=...)` |
| Solver | Division by zero | Guarded in analytical.py |
| SymPy | Solver failure | Try/except, display warning |
| Visualization | Plot error | Return None, skip plot |
| Verification | Any error | Return VerificationResult with error message |
