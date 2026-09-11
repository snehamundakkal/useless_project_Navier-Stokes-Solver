# Testing — LLM-Powered Navier–Stokes Solver

## Test Cases

### TEST 1 — Plane Poiseuille Flow

**Input:**
> "Water with dynamic viscosity 0.001 Pa·s and density 1000 kg/m³ flows between two stationary parallel plates separated by 0.02 m. The pressure gradient is -100 Pa/m. Find the velocity profile."

**Expected LLM Classification:** `poiseuille`

**Expected Analytical Results:**
- u_max = (100 × 0.02²) / (8 × 0.001) = **5.0 m/s** at y = h/2 = 0.01 m
- u_avg = (100 × 0.02²) / (12 × 0.001) = **3.333 m/s**
- Q/width = 3.333 × 0.02 = **0.06667 m²/s**
- Re = 1000 × 3.333 × 0.02 / 0.001 = **66,667** (turbulent — note this violates laminar assumption, so real flow would be turbulent; the formula still applies mathematically)

**Verification:** SymPy and NumPy should agree at y = 0.01 m to relative error < 1e-6.

**Expected Plot:** Parabola with maximum at y = 0.01 m.

---

### TEST 2 — Couette Flow

**Input:**
> "Oil with dynamic viscosity 0.01 Pa·s fills the gap between two parallel plates separated by 0.01 m. The upper plate moves at 0.5 m/s. No pressure gradient. Find the velocity distribution."

**Expected LLM Classification:** `couette`

**Expected Analytical Results:**
- u(y) = 0.5 × y / 0.01 = 50y m/s
- u(0) = 0 m/s (lower plate, no-slip)
- u(0.01) = 0.5 m/s (upper plate velocity ✓)
- u(0.005) = 0.25 m/s (midplane, linear profile)
- τ_w = 0.01 × 0.5 / 0.01 = **0.5 Pa**
- Re = 900 × 0.5 × 0.01 / 0.01 = **450** (laminar ✓)

**Verification:** Linear profile → SymPy and NumPy both give 0.25 m/s at y = h/2.

**Expected Plot:** Straight line from (0,0) to (0.5, 0.01).

---

### TEST 3 — Laminar Pipe Flow

**Input:**
> "Calculate the velocity profile for fully developed laminar flow of water (viscosity 0.001 Pa·s) through a circular pipe of radius 0.025 m. The pressure gradient is -200 Pa/m."

**Expected LLM Classification:** `pipe`

**Expected Analytical Results:**
- u_max = (200 × 0.025²) / (4 × 0.001) = **31.25 m/s**
- u_avg = u_max / 2 = **15.625 m/s**
- Q = π × 200 × 0.025⁴ / (8 × 0.001) = **0.03068 m³/s**
- Re = 1000 × 15.625 × 0.05 / 0.001 = **781,250** (turbulent — mathematically valid formula, flow assumption violated)

**Verification:** u(r = R/2) = (200/(4×0.001)) × (0.025² - 0.0125²) = 23.4375 m/s

**Expected Plot:** Parabolic profile + 2D colormap.

---

### TEST 4 — Reynolds Number

**Input:**
> "Calculate the Reynolds number for water (density 1000 kg/m³, viscosity 0.001 Pa·s) flowing through a pipe of radius 0.025 m at an average velocity of 1.0 m/s."

**Expected LLM Classification:** `reynolds`

**Expected Result:**
- L = D = 2 × 0.025 = 0.05 m
- Re = 1000 × 1.0 × 0.05 / 0.001 = **50,000** (turbulent)

**Expected Plot:** Regime gauge showing Re = 50,000 in turbulent zone.

---

### TEST 5 — Missing Parameter

**Input:**
> "Water flows between two parallel plates. Find the velocity profile."

**Expected Behavior:**
- LLM should flag missing parameters: plate separation h, pressure gradient or plate velocity
- `flags` list should be non-empty
- Solver should return `SolutionResult(success=False, error_message=...)`
- UI should display the error cleanly without crashing

---

### TEST 6 — Unsupported / Invalid Problem

**Input:**
> "Calculate the turbulent velocity profile in a 3D rectangular duct with wall roughness."

**Expected Behavior:**
- LLM should classify as `unknown` or flag as unsupported
- UI should display: "This problem type is not currently supported. Supported types: Poiseuille flow, Couette flow, Pipe flow, Reynolds number."
- No crash

---

### TEST 7 — No API Key / Demo Mode

**Setup:** Remove or blank `OPENAI_API_KEY` in `.env`.

**Expected Behavior:**
- App starts normally
- Yellow "DEMO MODE" banner appears
- Example problem buttons work
- Solve pipeline runs using `DEMO_RESPONSES` from `llm/client.py`
- All numerical results are computed normally
- Pre-written explanation is displayed
- Verification runs normally

---

### TEST 8 — Malformed LLM JSON

**Simulated input** (by temporarily mocking the LLM to return garbage):
```
Here is the analysis: The problem is about Poiseuille flow. mu=0.001, h=0.02...
```

**Expected Behavior:**
- `parser.py._extract_json_from_text()` fails to find a valid JSON object
- `parse_llm_response()` returns `(None, "JSON parse error: ...")`
- `extract_problem_parameters()` returns the error message
- App falls back to Demo Mode response
- User sees a warning (not a crash)

---

### TEST 9 — Numerical Verification

**For Poiseuille flow (TEST 1 parameters):**

SymPy symbolic at y = 0.01 m:
```python
u_expr = (1/(2*0.001)) * 100 * y * (0.02 - y)
u_sympy = u_expr.subs(y, 0.01) = 500 * 0.01 * 0.01 = 5.0
```

NumPy array at index 50 (y ≈ 0.01 m):
```python
u_numpy = (1/(2*0.001)) * 100 * 0.01 * (0.02 - 0.01) = 5.0
```

Relative error = |5.0 - 5.0| / 5.0 = **0.0** ✓ PASS

---

### TEST 10 — Graph Generation

**Expected for each problem type:**

| Type | Plot | Content |
|------|------|---------|
| Poiseuille | Parabola u(y) | Max at y = h/2, zero at walls |
| Couette | Straight line u(y) | Linear from 0 to U |
| Pipe | u(r) + colormap | Paraboloid, max at r=0 |
| Reynolds | Gauge bar | Re marker in correct regime |

**Expected for edge cases:**
- `result.success = False` → `generate_plot()` returns None → no crash
- `u_profile` all zeros (e.g., dpdx=0 Poiseuille) → flat line plotted

---

## Test Execution Summary

All tests were run with Demo Mode active (no API key required for core tests).

| Test | Status | Notes |
|------|--------|-------|
| 1 — Poiseuille | ✅ PASS | u_max = 5.0 m/s verified |
| 2 — Couette | ✅ PASS | Linear profile verified |
| 3 — Pipe | ✅ PASS | u_max = 31.25 m/s verified |
| 4 — Reynolds | ✅ PASS | Re = 50,000, turbulent |
| 5 — Missing param | ✅ PASS | Error displayed cleanly |
| 6 — Invalid problem | ✅ PASS | Unsupported message shown |
| 7 — Demo Mode | ✅ PASS | Full pipeline runs without API key |
| 8 — Malformed JSON | ✅ PASS | Parser handles gracefully |
| 9 — Verification | ✅ PASS | Rel. error < 1e-12 for all types |
| 10 — Graphs | ✅ PASS | All 4 plot types generated |
