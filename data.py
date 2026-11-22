"""Carga, limpieza y diagnóstico básico de los datos."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy import stats


def parse_text(text: str) -> np.ndarray:
    """Convierte texto pegado (una columna, filas o separado por ; , espacios) en un arreglo.

    Acepta coma decimal ("3,5") cuando hay varios valores en distintas líneas/espacios.
    """
    tokens = [t for t in re.split(r"[\s;]+", text.strip()) if t]
    if len(tokens) == 1 and "," in tokens[0]:          # "1,2,3,4" en una sola línea
        tokens = [t for t in tokens[0].split(",") if t]
    else:
        tokens = [t.replace(",", ".") for t in tokens]  # coma decimal
    try:
        x = np.array([float(t) for t in tokens])
    except ValueError as e:
        raise ValueError(f"No se pudo interpretar un valor numérico: {e}") from None
    return _clean(x)


def read_table(file) -> pd.DataFrame:
    name = getattr(file, "name", "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file, sep=None, engine="python")


def clean_series(s: pd.Series) -> np.ndarray:
    return _clean(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))


def _clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        raise ValueError("Se necesitan al menos 8 datos válidos.")
    return x


def is_discrete(x: np.ndarray) -> bool:
    return bool(np.all(np.equal(np.mod(x, 1), 0)) and x.min() >= 0)


def describe(x: np.ndarray) -> dict:
    return {
        "n": int(x.size), "min": x.min(), "max": x.max(), "media": x.mean(),
        "mediana": float(np.median(x)), "desv. est.": x.std(ddof=1),
        "varianza": x.var(ddof=1), "asimetría": float(stats.skew(x, bias=False)),
        "curtosis (exceso)": float(stats.kurtosis(x, bias=False)),
        "CV": x.std(ddof=1) / x.mean() if x.mean() != 0 else np.nan,
    }


def autocorr1(x: np.ndarray) -> float:
    d = x - x.mean()
    return float(np.sum(d[:-1] * d[1:]) / np.sum(d * d))


def runs_test(x: np.ndarray) -> tuple[float, float]:
    """Prueba de rachas (Wald-Wolfowitz) respecto a la mediana. Devuelve (z, p-valor).
    H0: los datos son independientes (aleatorios). Requiere el orden original."""
    med = np.median(x)
    s = x[x != med] > med
    n1, n2 = int(s.sum()), int((~s).sum())
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    n = n1 + n2
    mu = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    z = (runs - mu) / np.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))
