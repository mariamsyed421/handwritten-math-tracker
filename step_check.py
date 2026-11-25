# step_check.py: checks whether each step of a solution follows from the previous step
# and labels each step as OK, likely OCR error, or likely student error

from sympy import Eq, solve, Symbol
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
)

x = Symbol("x")
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
CONF_THRESHOLD = 0.7  # below this, we blame TrOCR instead of the student (tune on your data)


def parse_step(text):
    """Turn a cleaned line like '2x+4=10' into a sympy equation. Returns None if it can't."""
    text = text.replace("^", "**").replace("²", "**2").replace("−", "-")
    if text.count("=") != 1:
        return None  # a real equation step has exactly one '='
    left, right = text.split("=")
    try:
        eq = Eq(parse_expr(left, transformations=TRANSFORMS, local_dict={"x": x}),
                parse_expr(right, transformations=TRANSFORMS, local_dict={"x": x}))
    except Exception:
        return None
    # a real step should only contain x (e.g. "1905=0000" or "5x=L5" is a misread, not math)
    if eq is True or eq is False or getattr(eq, "free_symbols", None) != {x}:
        return None
    return eq


def solutions(eq):
    """Set of x values that satisfy the equation."""
    try:
        return set(solve(eq, x))
    except Exception:
        return None


def check_solution(lines, lowest_confs, threshold=CONF_THRESHOLD):
    """
    lines:        cleaned text of each step, in order, e.g. ["2x+4=10", "2x=6", "x=3"]
    lowest_confs: TrOCR's lowest token confidence for each step (from read_lines.py)
    returns:      one label per step
    """
    labels = []
    prev_sols = None
    for i, (text, conf) in enumerate(zip(lines, lowest_confs)):
        eq = parse_step(text)
        sols = solutions(eq) if eq is not None else None

        if sols is None:
            labels.append("OCR error (could not read as an equation)")
            continue  # keep comparing against the last readable step
        if prev_sols is None or sols == prev_sols:
            labels.append("OK")
        elif conf < threshold:
            labels.append(f"likely OCR error (inconsistent, low confidence {conf:.2f})")
            continue  # don't trust a misread step; compare the next step to the last good one
        else:
            labels.append(f"likely STUDENT error (inconsistent, high confidence {conf:.2f})")
        prev_sols = sols
    return labels