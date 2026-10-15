"""Shared number formatting for agent-fuzz papers (ra-01 #4861, 2026-10-14).

Rule: compute at full precision, round once at render time, ROUND_HALF_UP.
Values built from integer counts should be passed as fractions.Fraction (or int)
so that exact .5 boundaries (e.g. 33/80 = 0.4125) round up deterministically.
Floats are converted through their shortest repr, never through binary expansion.
"""
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction


def to_decimal(x):
    if isinstance(x, Decimal):
        return x
    if isinstance(x, (int, Fraction)):
        f = Fraction(x)
        return Decimal(f.numerator) / Decimal(f.denominator) if f.denominator != 1 else Decimal(f.numerator)
    if isinstance(x, float):
        return Decimal(repr(x))
    raise TypeError(f"unsupported type {type(x).__name__}")


def round_half_up(x, nd):
    """Round once to nd decimals with ROUND_HALF_UP; returns Decimal."""
    q = Decimal(1).scaleb(-nd)
    if isinstance(x, Fraction) and x.denominator != 1:
        # exact: scale, add 1/2 away from zero, floor
        s = x * Fraction(10) ** nd
        n = int(abs(s) + Fraction(1, 2))
        return (Decimal(n if s >= 0 else -n)).scaleb(-nd).quantize(q)
    return to_decimal(x).quantize(q, rounding=ROUND_HALF_UP)


def fmt(x, nd, pct=False):
    """Format x with nd decimals; pct=True multiplies by 100 exactly before rounding."""
    if pct:
        x = (Fraction(x) * 100) if not isinstance(x, float) else Decimal(repr(x)) * 100
    d = round_half_up(x, nd)
    if d == 0:
        d = abs(d)  # never print "-0.00"
    return f"{d:.{nd}f}"
