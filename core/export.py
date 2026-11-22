"""Exportación a simuladores (Arena, Simio, SimPy) y a Excel."""
from __future__ import annotations

import io
import numpy as np
import pandas as pd

from .fitting import FitResult, results_table, params_str


def _g(v: float) -> str:
    return f"{v:.6g}"


def _shift(expr: str, loc: float) -> str:
    return expr if abs(loc) < 1e-12 else f"{_g(loc)} + {expr}"


def _lognorm_mean_sd(s, scale):
    mean = scale * np.exp(s ** 2 / 2)
    return mean, mean * np.sqrt(np.exp(s ** 2) - 1)


def _gamma_parts(r: FitResult):
    """Distribuciones equivalentes a una gamma desplazada: (desplazamiento, signo, alfa, beta)."""
    p = r.params
    if r.key == "gamma":
        return p[1], 1, p[0], p[2]
    if r.key == "chi2":
        return p[1], 1, p[0] / 2, 2 * p[2]
    if r.key == "pearson3" and abs(p[0]) > 1e-6:
        sk, loc, sc = p
        a, th = 4 / sk ** 2, sc * abs(sk) / 2
        return (loc - 2 * sc / sk, 1, a, th) if sk > 0 else (loc + 2 * sc / abs(sk), -1, a, th)
    return None


def _gamma_expr(r, fn):
    shift, sign, a, b = _gamma_parts(r)
    e = fn(b, a)
    return _shift(e, shift) if sign > 0 else f"{_g(shift)} - {e}"


def arena_expr(r: FitResult) -> str | None:
    p, k = r.params, r.key
    if _gamma_parts(r):    return _gamma_expr(r, lambda b, a: f"GAMM({_g(b)}, {_g(a)})")
    if k == "rayleigh":    return _shift(f"WEIB({_g(p[1] * np.sqrt(2))}, 2)", p[0])
    if k == "norm":        return f"NORM({_g(p[0])}, {_g(p[1])})"
    if k == "expon":       return _shift(f"EXPO({_g(p[1])})", p[0])
    if k == "weibull_min": return _shift(f"WEIB({_g(p[2])}, {_g(p[0])})", p[1])
    if k == "lognorm":
        m, sd = _lognorm_mean_sd(p[0], p[2])
        return _shift(f"LOGN({_g(m)}, {_g(sd)})", p[1])
    if k == "uniform":     return f"UNIF({_g(p[0])}, {_g(p[0] + p[1])})"
    if k == "triang":      return f"TRIA({_g(p[1])}, {_g(p[1] + p[0] * p[2])}, {_g(p[1] + p[2])})"
    if k == "beta":        return f"{_g(p[2])} + {_g(p[3])} * BETA({_g(p[0])}, {_g(p[1])})"
    if k == "poisson":     return f"POIS({_g(p[0])})"
    return None


def simio_expr(r: FitResult) -> str | None:
    p, k = r.params, r.key
    if _gamma_parts(r):    return _gamma_expr(r, lambda b, a: f"Random.Gamma({_g(a)}, {_g(b)})")
    if k == "rayleigh":    return _shift(f"Random.Weibull(2, {_g(p[1] * np.sqrt(2))})", p[0])
    if k == "norm":        return f"Random.Normal({_g(p[0])}, {_g(p[1])})"
    if k == "expon":       return _shift(f"Random.Exponential({_g(p[1])})", p[0])
    if k == "weibull_min": return _shift(f"Random.Weibull({_g(p[0])}, {_g(p[2])})", p[1])
    if k == "lognorm":     return _shift(f"Random.Lognormal({_g(np.log(p[2]))}, {_g(p[0])})", p[1])
    if k == "uniform":     return f"Random.Uniform({_g(p[0])}, {_g(p[0] + p[1])})"
    if k == "triang":      return f"Random.Triangular({_g(p[1])}, {_g(p[1] + p[0] * p[2])}, {_g(p[1] + p[2])})"
    if k == "beta":        return f"{_g(p[2])} + {_g(p[3])} * Random.Beta({_g(p[0])}, {_g(p[1])})"
    if k == "poisson":     return f"Random.Poisson({_g(p[0])})"
    return None


def simpy_code(r: FitResult) -> str:
    args = ", ".join(f"{v:.8g}" for v in r.params)
    return (
        "from scipy import stats\n\n"
        f"dist = stats.{r.key}({args})   # {r.label}\n"
        "# En SimPy:  yield env.timeout(dist.rvs())\n"
        "muestra = dist.rvs(size=1000)\n"
    )


def excel_bytes(data: np.ndarray, results: list[FitResult]) -> bytes:
    table = results_table(results)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"datos": data}).to_excel(xw, sheet_name="Datos", index=False)
        table.to_excel(xw, sheet_name="Ranking", index=False)
        rows = [{"distribucion": r.label, "arena": arena_expr(r) or "—",
                 "simio": simio_expr(r) or "—", "parametros": params_str(r.key, r.params)}
                for r in results]
        pd.DataFrame(rows).to_excel(xw, sheet_name="Simuladores", index=False)
    return buf.getvalue()


def with_fallback(results, r: FitResult, fn):
    """Expresión de r; si no existe, la del mejor ajuste del ranking que sí la tenga.
    Devuelve (expresión, etiqueta_del_sustituto o None)."""
    e = fn(r)
    if e:
        return e, None
    for x in results:
        e = fn(x)
        if e:
            return e, x.label
    return None, None
