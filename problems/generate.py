from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from test_cases import glued, x  # noqa: E402

N_PER_TEMPLATE = 100
CHECK_GRID = np.linspace(-20.0, 20.0, 8001)
FIELDS = ["id", "family", "template", "param", "value",
          "a", "b", "c", "lambda", "f", "g", "a_min", "delta"]


def num(v: float):
    """
    Число с четырьмя значащими цифрами в виде sympy-Float.
    """
    return sp.Float(float(f"{v:.4g}"))


R = sp.Rational
pi = sp.pi
f1 = sp.exp(-x**2)

# Шаблон: (имя, параметр, значения, функция p -> (a, b, c, lambda, f)).
TEMPLATES = {
    "constant": [
        ("gauss_width", "k", np.linspace(0.5, 8.0, N_PER_TEMPLATE),
         lambda p: (1, 0, 0, 2, sp.exp(-p * x**2))),
        ("drift", "beta", np.linspace(-3.0, 3.0, N_PER_TEMPLATE),
         lambda p: (1, p, R(1, 2), 3, sp.cos(2 * x) * sp.exp(-2 * x**2))),
    ],
    "variable": [
        ("a_amplitude", "alpha", np.linspace(0.0, 0.8, N_PER_TEMPLATE),
         lambda p: (1 + p * sp.sin(pi * x), R(2, 5) * sp.cos(pi * x),
                    R(3, 10) * sp.tanh(x), R(3, 2),
                    sp.exp(-3 * x**2) * sp.sin(2 * x))),
        ("a_pinch", "alpha", np.linspace(0.0, 0.45, N_PER_TEMPLATE),
         lambda p: (R(1, 2) + p * sp.cos(2 * pi * x), sp.tanh(x) / 5, 0, 1,
                    sp.cos(3 * x) * sp.exp(-x**2))),
    ],
    "stiffness": [
        ("const_eps", "eps", np.geomspace(1e-3, 1e-1, N_PER_TEMPLATE),
         lambda p: (p, 2, 0, 1, sp.exp(-4 * x**2))),
        ("variable_eps", "eps", np.geomspace(1e-3, 1e-1, N_PER_TEMPLATE),
         lambda p: (p * (2 + sp.cos(pi * x)), 3 + sp.sin(pi * x), 0, 1,
                    sp.cos(4 * x) * sp.exp(-2 * x**2))),
    ],
    "oscillation": [
        ("solution_freq", "omega", np.linspace(5.0, 30.0, N_PER_TEMPLATE),
         lambda p: (1, 0, 0, 5, sp.cos(p * x) * sp.exp(-x**2))),
        ("c_freq", "omega", np.linspace(2.0, 30.0, N_PER_TEMPLATE),
         lambda p: (1, 0, 4 * sp.sin(p * x), 5, sp.exp(-2 * x**2))),
    ],
    "glued": [
        # склейка C^3: g непрерывна вместе с первой производной
        ("smooth_tail_freq", "omega", np.linspace(1.0, 10.0, N_PER_TEMPLATE),
         lambda p: (1, R(3, 10), R(1, 5), 2,
                    glued(f1, R(3, 10) * sp.cos(p * x), match_order=3))),
        # склейка C^1: g разрывна в нуле
        ("kink_tail_freq", "omega", np.linspace(1.0, 10.0, N_PER_TEMPLATE),
         lambda p: (1, 0, 0, 2,
                    glued(f1, -R(1, 2) * sp.cos(p * x), match_order=1))),
    ],
}


def vectorized(expr):
    fn = sp.lambdify(x, expr, modules="numpy")
    return lambda xs: np.broadcast_to(np.asarray(fn(xs), dtype=float),
                                      xs.shape)


def make_row(idx, family, template, param, value):
    p = num(value)
    a, b, c, lam, f = (sp.sympify(e) for e in TEMPLATES_FN[(family, template)](p))
    g = -(a * sp.diff(f, x, 2) + b * sp.diff(f, x) + (c - lam) * f)
    a_vals = vectorized(a)(CHECK_GRID)
    c_vals = vectorized(c)(CHECK_GRID)
    a_min = float(a_vals.min())
    delta = float(lam) - float(np.abs(c_vals).max())
    if a_min <= 0 or delta <= 0:
        raise ValueError(f"{family}/{template} {param}={value}: "
                         f"a_min={a_min:.3g}, delta={delta:.3g}")
    return dict(id=idx, family=family, template=template, param=param,
                value=f"{float(p):.4g}", a=sp.sstr(a), b=sp.sstr(b),
                c=sp.sstr(c), **{"lambda": sp.sstr(lam)}, f=sp.sstr(f),
                g=sp.sstr(g), a_min=f"{a_min:.4g}", delta=f"{delta:.4g}")


TEMPLATES_FN = {(fam, name): fn for fam, items in TEMPLATES.items()
                for name, _, _, fn in items}


def load(path):
    """
    Читает CSV в список словарей с sympy-выражениями (символ x).
    """
    out = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            loc = {"x": x}
            row = dict(row)
            for key in ("a", "b", "c", "lambda", "f", "g"):
                row[key] = sp.sympify(row[key], locals=loc)
            out.append(row)
    return out


def main():
    for family, items in TEMPLATES.items():
        rows = []
        for name, param, values, _ in items:
            for v in values:
                rows.append(make_row(len(rows) + 1, family, name, param, v))
        path = HERE / f"{family}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"{path.name}: {len(rows)} задач")


if __name__ == "__main__":
    main()
