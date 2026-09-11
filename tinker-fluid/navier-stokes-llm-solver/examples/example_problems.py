"""
examples/example_problems.py
=============================
Pre-defined example problems for the solver — used in Demo Mode and the UI cards.
"""

from typing import List
from dataclasses import dataclass


@dataclass
class ExampleProblem:
    id: str
    title: str
    description: str
    problem_text: str
    category: str   # "poiseuille" | "couette" | "pipe" | "reynolds"
    icon: str


EXAMPLE_PROBLEMS: List[ExampleProblem] = [
    ExampleProblem(
        id="ex1",
        title="Plane Poiseuille Flow",
        description="Pressure-driven viscous flow between two stationary plates",
        problem_text=(
            "Water with dynamic viscosity 0.001 Pa·s and density 1000 kg/m³ "
            "flows between two stationary parallel plates separated by 0.02 m. "
            "The pressure gradient is -100 Pa/m (driving flow in the positive x-direction). "
            "Find the velocity profile, maximum velocity, average velocity, and volumetric flow rate."
        ),
        category="poiseuille",
        icon="🔵",
    ),
    ExampleProblem(
        id="ex2",
        title="Couette Flow",
        description="Flow between plates where upper plate moves at constant velocity",
        problem_text=(
            "Oil with dynamic viscosity 0.01 Pa·s and density 900 kg/m³ fills the gap "
            "between two parallel plates separated by 0.01 m. "
            "The lower plate is stationary and the upper plate moves at 0.5 m/s. "
            "There is no applied pressure gradient. "
            "Find the velocity distribution and wall shear stress."
        ),
        category="couette",
        icon="🟠",
    ),
    ExampleProblem(
        id="ex3",
        title="Laminar Pipe Flow",
        description="Hagen-Poiseuille flow in a circular pipe",
        problem_text=(
            "Calculate the velocity profile for fully developed laminar flow of water "
            "(viscosity 0.001 Pa·s, density 1000 kg/m³) through a circular pipe of radius 0.025 m. "
            "The pressure gradient is -200 Pa/m. "
            "Find the maximum velocity, average velocity, and volumetric flow rate."
        ),
        category="pipe",
        icon="🟢",
    ),
    ExampleProblem(
        id="ex4",
        title="Reynolds Number",
        description="Determine flow regime for pipe flow",
        problem_text=(
            "Calculate the Reynolds number for water (density 1000 kg/m³, viscosity 0.001 Pa·s) "
            "flowing through a circular pipe of radius 0.025 m at an average velocity of 1.0 m/s. "
            "Determine whether the flow is laminar, transitional, or turbulent."
        ),
        category="reynolds",
        icon="🔴",
    ),
]


def get_example_by_id(eid: str) -> ExampleProblem | None:
    for ex in EXAMPLE_PROBLEMS:
        if ex.id == eid:
            return ex
    return None


def get_examples_by_category(category: str) -> List[ExampleProblem]:
    return [ex for ex in EXAMPLE_PROBLEMS if ex.category == category]
