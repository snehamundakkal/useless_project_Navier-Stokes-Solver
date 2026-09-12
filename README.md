<img width="1280" height="640" alt="git (1)" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />


# TinkerFluid 🌊 — LLM Navier–Stokes Solver

> *Because apparently solving century-old unsolved math problems with a chatbot was a totally necessary use of everyone's time.*

## Basic Details
### Team Name: OLISE


### Team Members
- Team Lead: Sneha Nair - NSS College of Engineering Palakkad
- Member 2: Stephin Arnold Raj S - NSS College of Engineering Palakkad

### Project Description
TinkerFluid is a Streamlit web app that lets you type a fluid mechanics problem in plain English, and then uses an LLM (OpenAI GPT) to extract the physics parameters — and a deterministic Python/SymPy/NumPy engine to actually solve it. It covers Poiseuille flow, Couette flow, pipe flow, and Reynolds number calculations, complete with symbolic derivations, numerical verification, and velocity profile visualizations. Dark mode and light mode included, because science should look good.

### The Problem (that doesn't exist)
Fluid mechanics students have been perfectly capable of solving Navier–Stokes equations by hand for centuries. Nobody asked for an AI to do it for them. The equations are on every textbook. The math is not that hard (for the special cases, anyway). Yet here we are.

### The Solution (that nobody asked for)
We built a system where you can ask "what happens when water flows between two plates at -100 Pa/m pressure gradient?" and an LLM reads your vibes, extracts the numbers, hands them off to SymPy to do the actual calculus (the LLM is NOT allowed to touch the math), NumPy verifies the answer, and then the LLM comes back to explain what just happened — like a PhD student who delegates all the hard work and then takes credit for it.

## Technical Details
### Technologies/Components Used
For Software:
- **Languages:** Python 3.10
- **Framework:** Streamlit 1.63 (web UI), with custom CSS theming (dark/light mode)
- **Libraries:** NumPy, SymPy, SciPy, Matplotlib, Pandas, OpenAI SDK, python-dotenv
- **Tools:** Git, pip, VS Code

### Implementation
For Software:

#### Installation
```bash
# Clone the repository
git clone https://github.com/snehamundakkal/useless_project_Navier-Stokes-Solver.git
cd useless_project_Navier-Stokes-Solver/tinker-fluid/navier-stokes-llm-solver

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...your-key...
```

#### Run
```bash
streamlit run app.py
# App opens at http://localhost:8501
# No API key? Toggle "Force Demo Mode" in the sidebar — it works fully without one.
```

#### System Architecture
```
User types a fluid mechanics problem in plain English
              ↓
         LLM (GPT-4o-mini)
     ├── Reads the problem
     ├── Extracts: μ, ρ, geometry, BCs
     └── Returns structured JSON
              ↓
      Python Engine (deterministic)
     ├── SymPy  → solves the ODE symbolically
     ├── NumPy  → evaluates the velocity profile numerically
     └── Matplotlib → plots the velocity profile
              ↓
      Verification Module
     └── SymPy vs NumPy cross-check (tolerance 1×10⁻⁶)
              ↓
         LLM (GPT-4o-mini)
     └── Explains the physics in plain English
              ↓
         Streamlit UI
     └── Displays everything beautifully
```

### Project Documentation
For Software:

#### Screenshots

![Main UI - Dark Mode](images/)
*The main interface in dark mode — hero title with animated gradient, example problem cards, and the problem input area*

#### Architecture Diagram
```
┌─────────────────────────────────────────────────────────┐
│                   STREAMLIT FRONTEND                     │
│  Dark/Light Theme │ Example Cards │ Problem Input        │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   LLM CLIENT        │
              │  (llm/client.py)    │
              │  OpenAI GPT API     │
              │  or Demo Mode       │
              └──────────┬──────────┘
                         │ JSON params
              ┌──────────▼──────────┐
              │   SOLVER ENGINE     │
              │  solver/            │
              │  ├─ analytical.py   │
              │  ├─ equations.py    │
              │  └─ validation.py   │
              └──────────┬──────────┘
                         │ SolutionResult
              ┌──────────▼──────────┐
              │  VISUALIZATION      │
              │  visualization/     │
              │  Matplotlib plots   │
              └─────────────────────┘
```

#### Supported Problem Types

| Flow Type | What it models | Governing Equation |
|-----------|---------------|-------------------|
| **Poiseuille** | Pressure-driven flow between stationary plates | `μ d²u/dy² = dp/dx` |
| **Couette** | Flow driven by a moving upper plate | `d²u/dy² = 0` |
| **Pipe Flow** | Hagen-Poiseuille flow in a circular pipe | `(1/r) d/dr(r du/dr) = dp/dz / μ` |
| **Reynolds** | Flow regime classification (laminar/transitional/turbulent) | `Re = ρVL/μ` |

#### Example Problems You Can Try
```
1. "Water (μ=0.001 Pa·s) flows between two stationary plates separated by
   0.02 m. Pressure gradient is -100 Pa/m. Find the velocity profile."

2. "Oil (μ=0.01 Pa·s, ρ=900 kg/m³) between two plates (h=0.01 m).
   Top plate moves at 0.5 m/s. No pressure gradient. Find velocity."

3. "Water flows through a circular pipe of radius 0.025 m.
   Pressure gradient = -200 Pa/m, μ = 0.001 Pa·s."

4. "What is the Reynolds number for water flowing at 1 m/s through
   a pipe of diameter 0.05 m?"
```

### Project Demo
#### Video
[Demo video coming soon]
*Full walkthrough of entering a fluid problem, watching the LLM extract parameters, the solver compute results, and the verification system confirm accuracy.*

#### Additional Demos
- Try it live at: http://localhost:8501 (run locally)
- Demo Mode works without any API key — just toggle it in the sidebar

## Team Contributions
- **Sneha Nair:** App architecture, solver integration,SymPy equation solver, numerical validation Streamlit UI design, dark/light theme system
- **Stephin Arnold Raj S:** LLM client integration, prompt engineering, parameter extraction pipeline, verification module, visualization module, PINN experiments


---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--26-26?link=https%3A%2F%2Ftinkerhub.org%2Fevents%2F1M8ORET9A1%2Fuseless-projects-3.0)



