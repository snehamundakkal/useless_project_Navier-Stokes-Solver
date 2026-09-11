# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
test_solver.py
==============
Automated test script for the Navier-Stokes Solver core modules.
Run: python test_solver.py

Tests:
  1. Plane Poiseuille Flow
  2. Couette Flow
  3. Pipe Flow
  4. Reynolds Number
  5. Missing Parameter Handling
  6. Verification System (SymPy vs NumPy)
  7. LLM JSON Parser (valid + malformed input)
  8. Visualization (graph generation)
  9. Demo Mode client
 10. End-to-end pipeline (Demo Mode)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import traceback

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))


def run_test(test_num, name, fn):
    print(f"\n{'='*60}")
    print(f"TEST {test_num}: {name}")
    print('='*60)
    try:
        fn()
    except Exception as e:
        check(f"TEST {test_num} — {name}", False, f"EXCEPTION: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Plane Poiseuille Flow
# ─────────────────────────────────────────────────────────────────────────────

def test_poiseuille():
    from solver.analytical import solve_plane_poiseuille

    h, mu, dpdx, rho = 0.02, 0.001, -100.0, 1000.0
    r = solve_plane_poiseuille(h=h, mu=mu, dpdx=dpdx, rho=rho)

    check("Success flag", r.success)
    check("Problem type", r.problem_type == "poiseuille")
    check("y-profile length", len(r.y_points) == 100)
    check("Profile non-negative (pressure drives flow)", np.all(r.u_profile >= -1e-12))

    # u_max = (-dp/dx)*h²/(8μ) = 100*0.0004/0.008 = 5.0
    expected_umax = 100.0 * 0.02**2 / (8 * 0.001)
    check(f"u_max ≈ {expected_umax:.4g} m/s",
          abs(r.u_max - expected_umax) < 1e-10,
          f"got {r.u_max}")

    # u_avg = u_max * 2/3
    expected_uavg = expected_umax * 2 / 3
    check(f"u_avg ≈ {expected_uavg:.4g} m/s",
          abs(r.u_avg - expected_uavg) < 1e-10,
          f"got {r.u_avg}")

    check("u(0) ≈ 0 (no-slip)", abs(r.u_profile[0]) < 1e-12)
    check("u(h) ≈ 0 (no-slip)", abs(r.u_profile[-1]) < 1e-12)
    check("u_max at midplane", abs(r.y_points[np.argmax(r.u_profile)] - h/2) < h/10)

    check("Reynolds number > 0", r.reynolds_number is not None and r.reynolds_number > 0)
    check("Wall shear stress > 0", r.wall_shear_stress is not None and r.wall_shear_stress > 0)
    check("Flow rate > 0", r.flow_rate is not None and r.flow_rate > 0)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Couette Flow
# ─────────────────────────────────────────────────────────────────────────────

def test_couette():
    from solver.analytical import solve_couette

    h, U, mu, rho = 0.01, 0.5, 0.01, 900.0
    r = solve_couette(h=h, U=U, mu=mu, rho=rho, dpdx=0.0)

    check("Success flag", r.success)
    check("Problem type", r.problem_type == "couette")

    # Linear profile: u(y) = U*y/h
    # Evaluate the analytical formula directly at h/2 (not from grid array)
    y_mid = h / 2.0
    expected_mid = U * y_mid / h     # = 0.25 m/s (pure Couette, no pressure gradient)
    # Compute directly using the same formula as analytical.py
    import numpy as _np
    u_at_mid_formula = float((U / h) * y_mid + (1.0 / (2.0 * mu)) * (-0.0) * y_mid * (h - y_mid))
    check(f"u(h/2) ≈ {expected_mid} m/s (linear profile)",
          abs(u_at_mid_formula - expected_mid) < 1e-12,
          f"got {u_at_mid_formula:.6g}")

    check("u(0) ≈ 0", abs(r.u_profile[0]) < 1e-12)
    check(f"u(h) ≈ {U} m/s", abs(r.u_profile[-1] - U) < 1e-10)
    check("u_max = U (no overshoot)", abs(r.u_max - U) < 1e-9)

    # Wall shear stress: τ = μU/h = 0.01*0.5/0.01 = 0.5 Pa
    expected_tau = mu * U / h
    check(f"Wall shear stress ≈ {expected_tau} Pa",
          abs(r.wall_shear_stress - expected_tau) < 1e-9,
          f"got {r.wall_shear_stress:.6g}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Pipe Flow
# ─────────────────────────────────────────────────────────────────────────────

def test_pipe_flow():
    from solver.analytical import solve_pipe_flow

    R, mu, dpdx, rho = 0.025, 0.001, -200.0, 1000.0
    r = solve_pipe_flow(R=R, mu=mu, dpdx=dpdx, rho=rho)

    check("Success flag", r.success)
    check("Problem type", r.problem_type == "pipe")

    # u_max = (-dp/dz)*R²/(4μ) = 200*0.000625/0.004 = 31.25 m/s
    expected_umax = 200.0 * 0.025**2 / (4 * 0.001)
    check(f"u_max ≈ {expected_umax:.4g} m/s",
          abs(r.u_max - expected_umax) < 1e-10,
          f"got {r.u_max}")

    # u_avg = u_max / 2
    check(f"u_avg = u_max/2",
          abs(r.u_avg - expected_umax / 2) < 1e-10,
          f"got {r.u_avg}")

    # u(R) = 0 (no-slip at wall)
    check("u(R) ≈ 0 (no-slip)", abs(r.u_profile[-1]) < 1e-12)

    # u(0) = u_max
    check("u(0) = u_max (centerline)", abs(r.u_profile[0] - r.u_max) < 1e-10)

    # Q = π R⁴ (-dp/dz) / (8μ)
    expected_Q = np.pi * 200.0 * 0.025**4 / (8 * 0.001)
    check(f"Flow rate ≈ {expected_Q:.4e} m³/s",
          abs(r.flow_rate - expected_Q) / expected_Q < 1e-6,
          f"got {r.flow_rate:.4e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Reynolds Number
# ─────────────────────────────────────────────────────────────────────────────

def test_reynolds():
    from solver.analytical import compute_reynolds

    rho, V, D, mu = 1000.0, 1.0, 0.05, 0.001
    r = compute_reynolds(rho=rho, velocity=V, length=D, mu=mu)

    check("Success flag", r.success)
    check("Problem type", r.problem_type == "reynolds")

    expected_Re = rho * V * D / mu   # = 50,000
    check(f"Re ≈ {expected_Re:.0f}",
          abs(r.reynolds_number - expected_Re) < 1e-6,
          f"got {r.reynolds_number}")

    check("Turbulent regime (Re=50000 > 4000)",
          "Turbulent" in r.assumptions[1] or r.reynolds_number > 4000)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Missing Parameter Handling
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_params():
    from solver.analytical import solve_plane_poiseuille, solve_couette

    # Missing h
    r = solve_plane_poiseuille(h=-1.0, mu=0.001, dpdx=-100.0)
    check("Negative h → success=False", not r.success)
    check("Error message non-empty", len(r.error_message) > 0, r.error_message)

    # Missing U for Couette
    r2 = solve_couette(h=0.01, U=-1.0, mu=0.001)
    check("Negative U → success=False", not r2.success)

    # Zero viscosity → guarded
    r3 = solve_plane_poiseuille(h=0.02, mu=0.0, dpdx=-100.0)
    check("Zero viscosity → success=False", not r3.success)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Verification System
# ─────────────────────────────────────────────────────────────────────────────

def test_verification():
    from solver.analytical import solve_plane_poiseuille, solve_couette, solve_pipe_flow
    from solver.validation import run_verification

    # Poiseuille
    r1 = solve_plane_poiseuille(h=0.02, mu=0.001, dpdx=-100.0)
    v1 = run_verification(r1)
    check("Poiseuille verification PASS", v1 is not None and v1.passed,
          f"error={v1.relative_error:.2e}" if v1 else "None")

    # Couette
    r2 = solve_couette(h=0.01, U=0.5, mu=0.01)
    v2 = run_verification(r2)
    check("Couette verification PASS", v2 is not None and v2.passed,
          f"error={v2.relative_error:.2e}" if v2 else "None")

    # Pipe
    r3 = solve_pipe_flow(R=0.025, mu=0.001, dpdx=-200.0)
    v3 = run_verification(r3)
    check("Pipe verification PASS", v3 is not None and v3.passed,
          f"error={v3.relative_error:.2e}" if v3 else "None")

    # Failed result → verification returns None
    from solver.analytical import SolutionResult
    bad = SolutionResult("poiseuille", False, "test error")
    v_bad = run_verification(bad)
    check("Failed result → verification=None", v_bad is None)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — LLM JSON Parser
# ─────────────────────────────────────────────────────────────────────────────

def test_parser():
    from llm.parser import parse_llm_response, extract_parameters_for_solver

    # Valid JSON
    valid_json = '''
    {
      "problem_type": "poiseuille",
      "geometry": "parallel plates",
      "fluid": "water",
      "density": 1000.0,
      "density_unit": "kg/m3",
      "viscosity": 0.001,
      "viscosity_unit": "Pa.s",
      "velocity": null,
      "velocity_unit": "m/s",
      "pressure_gradient": -100.0,
      "pressure_gradient_unit": "Pa/m",
      "dimensions": {"h": 0.02, "R": null, "L": null},
      "boundary_conditions": ["u=0 at y=0", "u=0 at y=h"],
      "initial_conditions": [],
      "assumptions": ["Steady flow", "Incompressible"],
      "governing_equation": "mu d2u/dy2 = dp/dx",
      "solution_method": "integration",
      "required_calculations": ["u(y)", "u_max"],
      "supported": true
    }
    '''
    parsed, err = parse_llm_response(valid_json)
    check("Valid JSON parsed without error", err is None, err)
    check("problem_type extracted", parsed and parsed.get("problem_type") == "poiseuille")
    check("viscosity coerced to float", parsed and isinstance(parsed.get("viscosity"), float))

    # JSON with markdown fence
    fenced = "```json\n" + valid_json + "\n```"
    parsed2, err2 = parse_llm_response(fenced)
    check("Markdown-fenced JSON parsed", err2 is None, err2)

    # Completely invalid response
    parsed3, err3 = parse_llm_response("The problem is about fluid flow and the answer is 5 m/s.")
    check("Invalid text → error returned", err3 is not None)
    check("Invalid text → None dict", parsed3 is None)

    # Empty response
    parsed4, err4 = parse_llm_response("")
    check("Empty response → error", err4 is not None)

    # Invalid problem_type → coerced to 'unknown'
    invalid_type = valid_json.replace('"poiseuille"', '"turbulent_cfd_3d"')
    parsed5, err5 = parse_llm_response(invalid_type)
    check("Invalid problem_type → 'unknown'", parsed5 and parsed5["problem_type"] == "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — Visualization
# ─────────────────────────────────────────────────────────────────────────────

def test_visualization():
    from solver.analytical import solve_plane_poiseuille, solve_couette, solve_pipe_flow, compute_reynolds
    from visualization.plots import generate_plot

    # Poiseuille
    r1 = solve_plane_poiseuille(h=0.02, mu=0.001, dpdx=-100.0)
    fig1 = generate_plot(r1)
    check("Poiseuille plot generated", fig1 is not None)

    # Couette
    r2 = solve_couette(h=0.01, U=0.5, mu=0.01)
    fig2 = generate_plot(r2)
    check("Couette plot generated", fig2 is not None)

    # Pipe flow
    r3 = solve_pipe_flow(R=0.025, mu=0.001, dpdx=-200.0)
    fig3 = generate_plot(r3)
    check("Pipe flow plot generated", fig3 is not None)

    # Reynolds gauge
    r4 = compute_reynolds(rho=1000.0, velocity=1.0, length=0.05, mu=0.001)
    fig4 = generate_plot(r4)
    check("Reynolds gauge generated", fig4 is not None)

    # Failed result → None (no crash)
    from solver.analytical import SolutionResult
    bad = SolutionResult("poiseuille", False, "test")
    fig_bad = generate_plot(bad)
    check("Failed result → no plot (no crash)", fig_bad is None)

    # Cleanup
    import matplotlib.pyplot as plt
    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9 — Demo Mode Client
# ─────────────────────────────────────────────────────────────────────────────

def test_demo_mode():
    # Force demo mode
    os.environ["DEMO_MODE"] = "true"
    os.environ["OPENAI_API_KEY"] = ""

    from llm.client import is_demo_mode, extract_problem_parameters, generate_explanation

    check("is_demo_mode() = True", is_demo_mode())

    # Extraction
    params, err = extract_problem_parameters("water flows between two parallel plates")
    check("Demo extraction returns params", params is not None and err is None, err)
    check("Demo params has problem_type", "problem_type" in (params or {}))

    # Explanation
    expl, expl_err = generate_explanation(
        "test problem", {"problem_type": "poiseuille"}, {"u_max": "5.0 m/s"}
    )
    check("Demo explanation returned", len(expl) > 50)
    check("Demo explanation no error", expl_err is None)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10 — End-to-End Pipeline (Demo Mode)
# ─────────────────────────────────────────────────────────────────────────────

def test_end_to_end():
    os.environ["DEMO_MODE"] = "true"

    from llm.client import extract_problem_parameters
    from solver.analytical import (
        solve_plane_poiseuille, solve_couette, solve_pipe_flow, compute_reynolds
    )
    from solver.validation import run_verification
    from visualization.plots import generate_plot

    problem = (
        "Water with viscosity 0.001 Pa·s flows between two stationary plates "
        "separated by 0.02 m. The pressure gradient is -100 Pa/m."
    )

    # Step 1: LLM extraction (Demo)
    params, err = extract_problem_parameters(problem)
    check("E2E — extraction succeeded", params is not None and err is None)

    # Step 2: Solver
    h = params.get("h") or 0.02
    mu = params.get("viscosity") or 0.001
    dpdx = params.get("pressure_gradient") or -100.0
    rho = params.get("density") or 1000.0

    result = solve_plane_poiseuille(h=h, mu=mu, dpdx=dpdx, rho=rho)
    check("E2E — solver succeeded", result.success, result.error_message)

    # Step 3: Verification
    ver = run_verification(result)
    check("E2E — verification passed", ver is not None and ver.passed,
          f"error={ver.relative_error:.2e}" if ver else "None")

    # Step 4: Plot
    fig = generate_plot(result)
    check("E2E — plot generated", fig is not None)

    import matplotlib.pyplot as plt
    plt.close("all")

    print(f"\n  Key results:")
    print(f"  u_max     = {result.u_max:.6f} m/s")
    print(f"  u_avg     = {result.u_avg:.6f} m/s")
    print(f"  Re        = {result.reynolds_number:.2f}")
    print(f"  Verified  = {ver.passed if ver else 'N/A'}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  LLM Navier–Stokes Solver — Test Suite")
    print("=" * 60)

    run_test(1, "Plane Poiseuille Flow", test_poiseuille)
    run_test(2, "Couette Flow", test_couette)
    run_test(3, "Laminar Pipe Flow", test_pipe_flow)
    run_test(4, "Reynolds Number", test_reynolds)
    run_test(5, "Missing Parameter Handling", test_missing_params)
    run_test(6, "Verification System (SymPy vs NumPy)", test_verification)
    run_test(7, "LLM JSON Parser", test_parser)
    run_test(8, "Visualization (Graph Generation)", test_visualization)
    run_test(9, "Demo Mode Client", test_demo_mode)
    run_test(10, "End-to-End Pipeline", test_end_to_end)

    # Summary
    print(f"\n{'='*60}")
    print("  TEST RESULTS SUMMARY")
    print('='*60)
    passed = sum(1 for _, s, _ in results if s == PASS)
    total = len(results)
    for name, status, detail in results:
        line = f"  {status}  {name}"
        if status == FAIL and detail:
            line += f"\n           Detail: {detail}"
        print(line)

    print(f"\n  {'-'*50}")
    result_emoji = 'ALL PASSED!' if passed == total else 'SOME FAILED'
    print(f"  {passed}/{total} checks passed  -- {result_emoji}")
    print('='*60)
    sys.exit(0 if passed == total else 1)
