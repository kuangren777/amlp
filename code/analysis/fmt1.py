"""One rounding rule for every printed number (ra-01 #4861, af-man #4875): full precision, ROUND_HALF_UP once,
through fmtnum.py (shared agent-fuzz formatter, shipped next to this file). Floats that are ratios of counts are first recovered as exact Fractions, so that a value
such as 25/4 = 6.25 rounds to 6.3 even when float noise put it at 6.2499999999999996."""
import os
import re
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fmtnum  # noqa: E402


def exact(v):
    """Fraction equal to v up to float noise (denominator <= 10**6), else None."""
    r = Fraction(v).limit_denominator(10 ** 6)
    return r if abs(float(r) - v) <= 1e-9 * max(1.0, abs(v)) else None


def F(v, spec=".1f"):
    """Format like f"{v:{spec}}" for spec in (".Nf", "+.Nf"), with ROUND_HALF_UP on the exact value."""
    m = re.fullmatch(r"(\+?)\.(\d+)f", spec)
    assert m, spec
    v = float(v)
    r = exact(v)
    s = fmtnum.fmt(r if r is not None else v, int(m.group(2)))
    if m.group(1) and not s.startswith("-"):
        s = "+" + s
    return s
