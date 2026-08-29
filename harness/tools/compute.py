"""Numeric and symbolic verification tools. No LLM calls.

These exist because the drafting stage repeatedly asserts quantities — "the
final speed is 3.58 m/s", "the two expressions are the same" — and a language
model checking its own arithmetic by inspection is exactly the failure mode a
rubric is supposed to catch in someone else. Everything here is reproducible
from the repository: given the same arguments it returns the same answer with no
network and no model involved.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .base import Tool, ToolContext, ToolResult
from .sandbox import run_python

__all__ = ["PythonEvalTool", "CheckEquivalenceTool", "CheckUnitsTool"]


class PythonEvalTool(Tool):
    name = "python_eval"
    description = """
    Run a short Python snippet to compute or verify something: arithmetic, unit
    conversion, solving an equation, checking a derivation numerically. Use it
    whenever you are about to assert a numeric value or an algebraic identity.
    `math`, `statistics`, `fractions`, `decimal`, `re`, `json`, and (when
    installed) `numpy` and `sympy` are already imported. Print what you want to
    see, or leave a trailing expression. No network, no file access; the process
    is discarded after each call, so nothing persists between calls.
    """
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source to execute. Print results you need to read.",
            },
            "purpose": {
                "type": "string",
                "description": "One short line: what this computation is meant to establish.",
            },
        },
        "required": ["code"],
    }

    def __init__(self, *, timeout_s: float = 20.0, memory_mb: int = 1024) -> None:
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb

    async def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        code = str(kwargs.get("code") or "")
        outcome = await run_python(
            code, timeout_s=self.timeout_s, memory_mb=self.memory_mb
        )
        data: dict[str, Any] = {"stdout": outcome.stdout}
        if outcome.value is not None:
            data["value"] = outcome.value
        if not outcome.ok:
            data["error"] = outcome.error
            if outcome.traceback:
                data["traceback"] = outcome.traceback[-800:]
        if outcome.ok and not outcome.stdout.strip() and outcome.value is None:
            data["note"] = "the snippet ran but produced no output — add a print()"
        return ToolResult(
            ok=outcome.ok,
            data=data,
            error=None if outcome.ok else outcome.error,
            meta={
                "seconds": round(outcome.seconds, 3),
                "timed_out": outcome.timed_out,
                "exit_code": outcome.exit_code,
                "purpose": str(kwargs.get("purpose") or "")[:200],
                "code": code[:2000],
            },
        )


# ---------------------------------------------------------------------------
# Symbolic / numeric equivalence
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _sympy():
    try:
        import sympy  # noqa: PLC0415 - optional, checked at call time

        return sympy
    except Exception:  # noqa: BLE001
        return None


def _to_float(text: str) -> float | None:
    match = _NUM_RE.search(text.replace(",", ""))
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


class CheckEquivalenceTool(Tool):
    name = "check_equivalence"
    description = """
    Decide whether two mathematical expressions or numeric answers are the same.
    Handles algebraic rearrangement (`v*r/R` vs `r*v/R`), different but
    equivalent forms, and ordinary rounding (`3.58` vs `3.5777`). Use it before
    writing a criterion that pins down a value, so the criterion accepts every
    correct way of writing it instead of one particular spelling.
    """
    parameters = {
        "type": "object",
        "properties": {
            "left": {"type": "string", "description": "First expression or value, e.g. '3.58' or 'sqrt(2*g*h)'."},
            "right": {"type": "string", "description": "Second expression or value to compare against."},
            "tolerance": {
                "type": "number",
                "description": "Relative tolerance for the numeric comparison. Default 0.01 (1%).",
            },
        },
        "required": ["left", "right"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        left = str(kwargs.get("left") or "").strip()
        right = str(kwargs.get("right") or "").strip()
        try:
            tol = abs(float(kwargs.get("tolerance", 0.01)))
        except (TypeError, ValueError):
            tol = 0.01
        if not left or not right:
            return ToolResult.failure("both 'left' and 'right' must be non-empty")

        findings: list[str] = []
        symbolic: bool | None = None
        numeric: bool | None = None

        sp = _sympy()
        if sp is not None:
            try:
                le = sp.sympify(left, rational=True)
                re_ = sp.sympify(right, rational=True)
                diff = sp.simplify(le - re_)
                symbolic = bool(diff == 0)
                findings.append(
                    "symbolically identical" if symbolic
                    else f"symbolic difference simplifies to {diff}"
                )
                if not symbolic:
                    try:
                        lv, rv = complex(sp.N(le)), complex(sp.N(re_))
                        numeric = _close(lv.real, rv.real, tol) and _close(lv.imag, rv.imag, tol)
                    except (TypeError, ValueError, AttributeError):
                        pass
            except Exception as exc:  # noqa: BLE001 - sympify rejects prose readily
                findings.append(f"could not parse symbolically ({type(exc).__name__})")

        if numeric is None:
            lf, rf = _to_float(left), _to_float(right)
            if lf is not None and rf is not None:
                numeric = _close(lf, rf, tol)
                findings.append(
                    f"numerically {'equal' if numeric else 'different'} "
                    f"({lf:g} vs {rf:g}, tolerance {tol:.3g})"
                )

        if symbolic is None and numeric is None:
            return ToolResult(
                ok=True,
                data={
                    "equivalent": None,
                    "explanation": "neither expression could be parsed as maths; "
                                   "compare them yourself or use python_eval",
                    "findings": findings,
                },
            )

        equivalent = bool(symbolic or numeric)
        return ToolResult(
            ok=True,
            data={
                "equivalent": equivalent,
                "symbolically_equal": symbolic,
                "numerically_equal": numeric,
                "tolerance": tol,
                "explanation": "; ".join(findings) or "compared",
            },
            meta={"left": left[:300], "right": right[:300]},
        )


def _close(a: float, b: float, tol: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol * max(1.0, abs(a), abs(b)) * 1e-6 or 1e-12)


# ---------------------------------------------------------------------------
# Dimensional analysis
# ---------------------------------------------------------------------------

#: Base dimensions as (length, mass, time, current, temperature, amount).
_DIMS: dict[str, tuple[tuple[int, ...], float]] = {}


def _register(names: str, dims: tuple[int, ...], scale: float) -> None:
    for name in names.split():
        _DIMS[name] = (dims, scale)


_L, _M, _T = (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)
_ONE = (0, 0, 0, 0, 0, 0)

_register("m metre metres meter meters", _L, 1.0)
_register("km kilometre kilometres kilometer kilometers", _L, 1e3)
_register("cm centimetre centimetres centimeter centimeters", _L, 1e-2)
_register("mm millimetre millimetres millimeter millimeters", _L, 1e-3)
_register("kg kilogram kilograms", _M, 1.0)
_register("g gram grams", _M, 1e-3)
_register("mg milligram milligrams", _M, 1e-6)
_register("s sec secs second seconds", _T, 1.0)
_register("ms millisecond milliseconds", _T, 1e-3)
_register("min minute minutes", _T, 60.0)
_register("h hr hour hours", _T, 3600.0)
_register("A amp amps ampere amperes", (0, 0, 0, 1, 0, 0), 1.0)
_register("K kelvin", (0, 0, 0, 0, 1, 0), 1.0)
_register("mol mole moles", (0, 0, 0, 0, 0, 1), 1.0)
_register("N newton newtons", (1, 1, -2, 0, 0, 0), 1.0)
_register("J joule joules", (2, 1, -2, 0, 0, 0), 1.0)
_register("kJ kilojoule kilojoules", (2, 1, -2, 0, 0, 0), 1e3)
_register("W watt watts", (2, 1, -3, 0, 0, 0), 1.0)
_register("Pa pascal pascals", (-1, 1, -2, 0, 0, 0), 1.0)
_register("kPa kilopascal kilopascals", (-1, 1, -2, 0, 0, 0), 1e3)
_register("C coulomb coulombs", (0, 0, 1, 1, 0, 0), 1.0)
_register("V volt volts", (2, 1, -3, -1, 0, 0), 1.0)
_register("Hz hertz", (0, 0, -1, 0, 0, 0), 1.0)
_register("L l litre litres liter liters", (3, 0, 0, 0, 0, 0), 1e-3)
_register("mL ml millilitre millilitres milliliter milliliters", (3, 0, 0, 0, 0, 0), 1e-6)
_register("rad radian radians degree degrees deg percent", _ONE, 1.0)

_DIM_NAMES = ("length", "mass", "time", "current", "temperature", "amount")
_UNIT_TOKEN_RE = re.compile(r"([A-Za-zµ°%]+)\s*(?:\^|\*\*)?\s*(-?\d+)?")


def _parse_unit(text: str) -> tuple[tuple[int, ...], float, list[str]] | None:
    """Very small unit-expression parser: handles ``kg*m/s^2``, ``m s^-1``, ``J``."""
    body = (text or "").strip()
    if not body:
        return None
    body = _NUM_RE.sub(" ", body, count=1) if _NUM_RE.match(body) else body
    body = body.replace("·", "*").replace("×", "*").replace("÷", "/")

    dims = [0] * 6
    scale = 1.0
    unknown: list[str] = []
    seen = False
    # Split on '/' so everything after the first solidus is inverted.
    for part_index, chunk in enumerate(body.split("/")):
        sign = 1 if part_index == 0 else -1
        for token, exponent in _UNIT_TOKEN_RE.findall(chunk):
            if not token:
                continue
            entry = _DIMS.get(token) or _DIMS.get(token.lower())
            power = int(exponent) if exponent else 1
            if entry is None:
                unknown.append(token)
                continue
            seen = True
            base, factor = entry
            for i, d in enumerate(base):
                dims[i] += sign * power * d
            scale *= factor ** (sign * power)
    if not seen:
        return None
    return tuple(dims), scale, unknown


class CheckUnitsTool(Tool):
    name = "check_units"
    description = """
    Check that a quantity's units are dimensionally consistent, and optionally
    that they match an expected unit. Give it something like '3.58 m/s' and
    expected 'm/s', or ask whether 'kg*m/s^2' and 'N' are the same dimension.
    Use it before writing a criterion that requires a particular unit, so the
    criterion does not demand a spelling the reference answer never uses.
    """
    parameters = {
        "type": "object",
        "properties": {
            "quantity": {
                "type": "string",
                "description": "The quantity or unit expression to check, e.g. '9.8 m/s^2' or 'kg*m/s^2'.",
            },
            "expected_unit": {
                "type": "string",
                "description": "Optional unit to compare against, e.g. 'N' or 'm/s'.",
            },
        },
        "required": ["quantity"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        quantity = str(kwargs.get("quantity") or "").strip()
        expected = str(kwargs.get("expected_unit") or "").strip()
        if not quantity:
            return ToolResult.failure("'quantity' must be non-empty")

        parsed = _parse_unit(quantity)
        if parsed is None:
            return ToolResult(
                ok=True,
                data={
                    "recognised": False,
                    "explanation": f"no known unit found in {quantity!r}; "
                                   "the checker knows SI units and common derived ones",
                },
            )
        dims, scale, unknown = parsed
        data: dict[str, Any] = {
            "recognised": True,
            "dimensions": {n: d for n, d in zip(_DIM_NAMES, dims) if d},
            "dimensionless": all(d == 0 for d in dims),
        }
        if unknown:
            data["unrecognised_tokens"] = sorted(set(unknown))

        if expected:
            exp = _parse_unit(expected)
            if exp is None:
                data["matches_expected"] = None
                data["explanation"] = f"could not parse expected unit {expected!r}"
            else:
                same = exp[0] == dims
                data["matches_expected"] = same
                data["expected_dimensions"] = {
                    n: d for n, d in zip(_DIM_NAMES, exp[0]) if d
                }
                if same:
                    ratio = scale / exp[1] if exp[1] else None
                    data["conversion_factor"] = ratio
                    data["explanation"] = (
                        "same dimension"
                        + (f"; 1 {quantity} = {ratio:g} {expected}" if ratio and ratio != 1.0 else "")
                    )
                else:
                    data["explanation"] = "DIMENSION MISMATCH — these are not the same kind of quantity"
        return ToolResult(ok=True, data=data, meta={"quantity": quantity[:200]})
