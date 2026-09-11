# Equations Reference — Navier–Stokes Solver

## General Incompressible Navier–Stokes Equations

### Momentum Equation
```
∂u/∂t + (u · ∇)u = -(1/ρ)∇p + ν∇²u + f
```

### Continuity Equation (Incompressibility)
```
∇ · u = 0
```

Where:
- **u** = velocity vector [m/s]
- **ρ** = density [kg/m³]
- **p** = pressure [Pa]
- **μ** = dynamic viscosity [Pa·s]
- **ν = μ/ρ** = kinematic viscosity [m²/s]
- **f** = body force per unit mass [m/s²]

---

## 1. Plane Poiseuille Flow

**Setup:** Pressure-driven flow between two stationary plates separated by h.

**Simplification steps:**
1. Steady flow → ∂u/∂t = 0
2. Fully developed → ∂u/∂x = 0, v = 0
3. Continuity → satisfied automatically

**Governing ODE:**
```
μ d²u/dy² = dp/dx
```

**Solution (with BCs u(0) = u(h) = 0):**
```
u(y) = (1/2μ)(-dp/dx) · y(h - y)
```

**Derived quantities:**
```
u_max = (-dp/dx)h²/(8μ)         [at y = h/2]
u_avg = (-dp/dx)h²/(12μ) = (2/3)u_max
Q     = u_avg × h                [per unit width, m²/s]
τ_w   = |dp/dx| × h/2           [wall shear stress, Pa]
Re    = ρ u_avg h / μ
```

---

## 2. Couette Flow

**Setup:** Upper plate moves at U, lower plate stationary. No pressure gradient.

**Governing ODE:**
```
μ d²u/dy² = 0
```

**Solution (with BCs u(0) = 0, u(h) = U):**
```
u(y) = (U/h) y
```

**Generalised Couette (with pressure gradient):**
```
u(y) = (U/h)y + (1/2μ)(-dp/dx) y(h - y)
```

**Derived quantities:**
```
τ_w = μ U/h          [constant throughout, Pa]
Re  = ρ U h / μ
```

---

## 3. Hagen-Poiseuille Pipe Flow

**Setup:** Fully developed laminar flow in a circular pipe of radius R.

**Governing ODE (cylindrical coordinates):**
```
(1/r) d/dr(r du_z/dr) = (1/μ) dp/dz
```

Equivalently:
```
d²u_z/dr² + (1/r) du_z/dr = (1/μ) dp/dz
```

**BCs:** du_z/dr = 0 at r = 0 (symmetry), u_z(R) = 0 (no-slip)

**Solution:**
```
u_z(r) = (1/4μ)(-dp/dz)(R² - r²)
```

**Derived quantities:**
```
u_max = (-dp/dz)R²/(4μ)          [at r = 0, centerline]
u_avg = (-dp/dz)R²/(8μ) = u_max/2
Q     = π R⁴(-dp/dz)/(8μ)        [Hagen-Poiseuille law, m³/s]
τ_w   = |dp/dz| R/2              [Pa]
Re    = ρ u_avg (2R) / μ
```

**Note:** The Q ∝ R⁴ dependence (doubling radius → 16× flow rate) is one of the most important results in viscous flow.

---

## 4. Reynolds Number

```
Re = ρVL/μ = VL/ν
```

Where V = characteristic velocity, L = characteristic length (h for plates, D = 2R for pipes).

**Flow regimes (pipe flow):**
```
Re < 2300         → Laminar
2300 ≤ Re < 4000  → Transitional
Re ≥ 4000         → Turbulent
```

**Significance:** Re is the ratio of inertial to viscous forces. The Hagen-Poiseuille and Poiseuille solutions are only valid in the laminar regime.

---

## 5. Derived Quantities

### Kinematic viscosity
```
ν = μ/ρ   [m²/s]
```

### Pressure drop
```
ΔP = |dp/dx| × L   [Pa]
```

### Darcy-Weisbach friction factor (laminar pipe)
```
f = 64/Re
```

### Hydrodynamic entry length
```
L_e ≈ 0.06 Re D   (laminar)
```

---

## Validity Domains

| Solution | Valid When |
|----------|------------|
| Poiseuille | Re < ~1000 (channel), fully developed, Newtonian, steady |
| Couette | Re < ~1000, fully developed, Newtonian, steady |
| Hagen-Poiseuille | Re < 2300, fully developed, Newtonian, steady, axisymmetric |
| All above | Incompressible (Ma << 1), constant viscosity, no body forces |
