"""
llm/parser.py
=============
JSON parser and validator for LLM structured output.
Validates and sanitizes the LLM response before it is passed to the solver.
"""

import json
import re
from typing import Tuple, Optional


# Expected problem types
VALID_PROBLEM_TYPES = {"poiseuille", "couette", "pipe", "reynolds", "unknown"}

# Required top-level keys in the JSON schema
REQUIRED_KEYS = {
    "problem_type", "geometry", "fluid", "density", "viscosity",
    "velocity", "pressure_gradient", "dimensions",
    "boundary_conditions", "assumptions", "governing_equation",
    "solution_method", "required_calculations", "supported",
}


def _extract_json_from_text(text: str) -> str:
    """
    Attempt to extract a JSON object from text that may contain
    markdown code fences or other surrounding content.
    """
    # Remove markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()

    # Try to find the first { ... } block
    start = text.find("{")
    if start == -1:
        return text

    # Find matching closing brace
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text[start:]


def parse_llm_response(raw_text: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Parse and validate the LLM JSON response.

    Returns:
        (parsed_dict, None)  on success
        (None, error_message) on failure
    """
    if not raw_text or not raw_text.strip():
        return None, "LLM returned an empty response."

    json_str = _extract_json_from_text(raw_text)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}. Raw snippet: {json_str[:200]}"

    if not isinstance(data, dict):
        return None, "LLM response is not a JSON object."

    # Validate problem_type
    ptype = data.get("problem_type", "unknown")
    if ptype not in VALID_PROBLEM_TYPES:
        data["problem_type"] = "unknown"
        data.setdefault("flags", [])
        data["flags"].append(f"Unrecognised problem_type '{ptype}'. Set to 'unknown'.")

    # Ensure lists exist
    for list_key in ("boundary_conditions", "initial_conditions", "assumptions",
                     "required_calculations", "flags"):
        if not isinstance(data.get(list_key), list):
            data[list_key] = []

    # Ensure dimensions dict exists
    if not isinstance(data.get("dimensions"), dict):
        data["dimensions"] = {}

    # Coerce numeric fields (can be null / string / number)
    for num_key in ("density", "viscosity", "velocity", "pressure_gradient"):
        val = data.get(num_key)
        if val is not None:
            try:
                data[num_key] = float(val)
            except (TypeError, ValueError):
                data[num_key] = None
                data["flags"].append(f"Could not parse '{num_key}' = {val!r} as a number.")

    # Coerce dimension values
    for dim_key in ("h", "R", "L"):
        val = data.get("dimensions", {}).get(dim_key)
        if val is not None:
            try:
                data["dimensions"][dim_key] = float(val)
            except (TypeError, ValueError):
                data["dimensions"][dim_key] = None

    # Ensure supported flag is bool
    if not isinstance(data.get("supported"), bool):
        data["supported"] = data.get("problem_type") != "unknown"

    # Add missing optional keys with defaults
    data.setdefault("problem_summary", "")
    data.setdefault("flags", [])
    data.setdefault("fluid", "unknown")

    return data, None


def extract_parameters_for_solver(parsed: dict) -> dict:
    """
    Flatten the parsed JSON into a flat dict suitable for the solver dispatch.
    Unit conversions could be added here in the future.
    """
    dims = parsed.get("dimensions", {})
    return {
        "problem_type": parsed.get("problem_type", "unknown"),
        "fluid": parsed.get("fluid", "unknown"),
        "density": parsed.get("density"),
        "viscosity": parsed.get("viscosity"),
        "velocity": parsed.get("velocity"),
        "pressure_gradient": parsed.get("pressure_gradient"),
        "h": dims.get("h"),
        "R": dims.get("R"),
        "L": dims.get("L"),
        "assumptions": parsed.get("assumptions", []),
        "boundary_conditions": parsed.get("boundary_conditions", []),
        "flags": parsed.get("flags", []),
        "supported": parsed.get("supported", False),
        "governing_equation": parsed.get("governing_equation", ""),
        "solution_method": parsed.get("solution_method", ""),
        "problem_summary": parsed.get("problem_summary", ""),
    }
