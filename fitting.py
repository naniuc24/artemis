"""Ajuste por máxima verosimilitud, criterios de información y bondad de ajuste."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# clave scipy -> (etiqueta, ¿soporte positivo?)
CONTINUOUS = {
    "norm": ("Normal", False), "lognorm": ("Lognormal", True),
    "expon": ("Exponencial", True), "gamma": ("Gamma", True),
    "weibull_min": ("Weibull", True), "beta": ("Beta", False),
    "triang": ("Triangular", False), "uniform": ("Uniforme", False),
    "logistic": ("Logística", False), "fisk": ("Log-logística", True),
    "invgauss": ("Gaussiana inversa", True), "rayleigh": ("Rayleigh", True),
    "gumbel_r": ("Gumbel (máximo)", False), "gumbel_l": ("Gumbel (mínimo)", False),
    "laplace": ("Laplace", False), "pearson3": ("Pearson III", False),
    "genextreme": ("Valor extremo generalizado", False), "pareto": ("Pareto", True),
    "powerlaw": ("Potencia", False), "nakagami": ("Nakagami", True),
    "chi2": ("Chi-cuadrada", True),
}
BOUNDED = {"beta", "triang", "powerlaw"}  # cotas estimadas de los datos (min y rango)
DISCRETE = {"poisson": "Poisson", "geom": "Geométrica", "nbinom": "Binomial negativa"}
LABELS = {**{k: v[0] for k, v in CONTINUOUS.items()}, **DISCRETE}


@dataclass
class FitResult:
    key: str
    label: str
    params: tuple
    k: int
    n: int
    discrete: bool
    loglik: float
    aic: float
    bic: float
    ks: float = np.nan
    ks_p: float = np.nan      # p-valor asintótico (referencial: parámetros estimados)
    ad: float = np.nan
    chi2: float = np.nan
    chi2_p: float = np.nan
    chi2_dof: int = 0

    @property
    def dist(self):
        return getattr(stats, self.key)

    def frozen(self):
        return self.dist(*self.params)


def param_names(key: str) -> list[str]:
    d = getattr(stats, key)
    shapes = [s.strip() for s in d.shapes.split(",")] if d.shapes else []
    return shapes if key in DISCRETE else shapes + ["loc", "scale"]


def params_str(key: str, params: tuple) -> str:
    return ", ".join(f"{n}={v:.5g}" for n, v in zip(param_names(key), params))


# ----------------------------------------------------------------- ajuste
def _fit_continuous(x, key, shift):
    dist = getattr(stats, key)
    positive = CONTINUOUS[key][1]
    lo, hi = float(x.min()), float(x.max())
    rng = (hi - lo) or 1.0
    if key == "uniform":
        return (lo, rng), 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if key in BOUNDED:
            p = dist.fit(x, floc=lo - 0.001 * rng, fscale=rng * 1.002)
            return p, len(p)
        if positive:
            if shift:  # parámetro de desplazamiento (loc) libre
                p = dist.fit(x, loc=lo - 0.01 * rng)
                if np.isfinite(dist.logpdf(x, *p)).all():
                    return p, len(p)
            if lo < 0:
                raise ValueError("Datos negativos: requiere desplazamiento.")
            p = dist.fit(x, floc=0)
            return p, len(p) - 1
        p = dist.fit(x)
        return p, len(p)


def _fit_discrete(x, key):
    m, v = x.mean(), x.var(ddof=0)
    if key == "poisson":
        return (m,), 1
    if key == "geom":
        if x.min() < 1:
            raise ValueError("Geométrica requiere valores >= 1.")
        return (1.0 / m,), 1
    if key == "nbinom":
        if v <= m:
            raise ValueError("Binomial negativa requiere varianza > media.")
        p = m / v
        return (m * p / (1 - p), p), 2     # método de momentos
    raise KeyError(key)


# ------------------------------------------------------- bondad de ajuste
def _ks_ad(x, dist, params):
    xs = np.sort(x)
    n = len(xs)
    F = np.clip(dist.cdf(xs, *params), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    ks = max(np.max(i / n - F), np.max(F - (i - 1) / n))
    ad = -n - np.mean((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1])))
    return float(ks), float(ad)


def _chi2_cont(x, dist, params, kpar):
    n = len(x)
    bins = int(max(4, min(np.ceil(1 + 3.322 * np.log10(n)), n / 5)))
    edges = dist.ppf(np.linspace(0, 1, bins + 1), *params)
    edges[0], edges[-1] = -np.inf, np.inf
    if not np.all(np.isfinite(edges[1:-1])) or np.any(np.diff(edges) <= 0):
        return np.nan, np.nan, 0
    obs = np.histogram(x, edges)[0]
    exp = n / bins
    stat = float(np.sum((obs - exp) ** 2 / exp))
    dof = bins - 1 - kpar
    return stat, (float(stats.chi2.sf(stat, dof)) if dof > 0 else np.nan), dof


def _chi2_disc(x, dist, params, kpar):
    n = len(x)
    ks_ = np.arange(int(x.min()), int(x.max()) + 1)
    obs = np.array([(x == v).sum() for v in ks_], float)
    exp = n * dist.pmf(ks_, *params)
    exp[0] += n * dist.cdf(ks_[0] - 1, *params)
    exp[-1] += n * dist.sf(ks_[-1], *params)
    O, E, co, ce = [], [], 0.0, 0.0
    for o, e in zip(obs, exp):          # agrupar clases con esperado < 5
        co += o; ce += e
        if ce >= 5:
            O.append(co); E.append(ce); co = ce = 0.0
    if ce > 0:
        if E:
            O[-1] += co; E[-1] += ce
        else:
            O.append(co); E.append(ce)
    O, E = np.array(O), np.array(E)
    stat = float(np.sum((O - E) ** 2 / E))
    dof = len(O) - 1 - kpar
    return stat, (float(stats.chi2.sf(stat, dof)) if dof > 0 else np.nan), dof


def fit_one(x, key, shift=False, light=False) -> FitResult:
    """Ajusta una distribución. light=True omite χ² (más rápido, para bootstrap)."""
    x = np.asarray(x, float)
    n = len(x)
    disc = key in DISCRETE
    params, k = _fit_discrete(x, key) if disc else _fit_continuous(x, key, shift)
    dist = getattr(stats, key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ll = float(np.sum(dist.logpmf(x, *params) if disc else dist.logpdf(x, *params)))
        if not np.isfinite(ll):
            raise ValueError("Verosimilitud no finita.")
        r = FitResult(key, LABELS[key], tuple(float(p) for p in params), k, n, disc,
                      ll, 2 * k - 2 * ll, k * np.log(n) - 2 * ll)
        if not disc:
            r.ks, r.ad = _ks_ad(x, dist, params)
            r.ks_p = float(stats.kstwo.sf(r.ks, n))
        if not light:
            f = _chi2_disc if disc else _chi2_cont
            r.chi2, r.chi2_p, r.chi2_dof = f(x, dist, params, k)
    return r


def fit_all(x, keys=None, shift=False) -> list[FitResult]:
    """Ajusta todas las distribuciones pedidas; descarta las que fallan. Ordena por AIC."""
    keys = keys or list(CONTINUOUS)
    out = []
    for key in keys:
        try:
            out.append(fit_one(x, key, shift))
        except Exception:
            continue
    return sorted(out, key=lambda r: r.aic)


def results_table(results: list[FitResult]) -> pd.DataFrame:
    df = pd.DataFrame([{
        "key": r.key, "label": r.label, "params": params_str(r.key, r.params), "k": r.k,
        "loglik": r.loglik, "AIC": r.aic, "BIC": r.bic, "KS": r.ks, "KS_p": r.ks_p,
        "AD": r.ad, "chi2": r.chi2, "chi2_p": r.chi2_p} for r in results])
    if df.empty:
        return df
    for col, new in [("KS", "rank_KS"), ("AD", "rank_AD"), ("chi2", "rank_chi2"), ("AIC", "rank_AIC")]:
        df[new] = df[col].rank(method="min")
    df["rank_mean"] = df[["rank_KS", "rank_AD", "rank_chi2", "rank_AIC"]].mean(axis=1)
    return df


def bootstrap_pvalues(x, res: FitResult, shift=False, n_boot=200, seed=0):
    """Bootstrap paramétrico (Monte Carlo): p-valores de KS y A-D válidos aunque los
    parámetros se hayan estimado de los mismos datos. Devuelve (p_ks, p_ad)."""
    if res.discrete:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    dist, n = res.dist, len(x)
    c_ks = c_ad = valid = 0
    for _ in range(n_boot):
        sim = dist.rvs(*res.params, size=n, random_state=rng)
        try:
            r = fit_one(sim, res.key, shift, light=True)
        except Exception:
            continue
        valid += 1
        c_ks += r.ks >= res.ks
        c_ad += r.ad >= res.ad
    if valid == 0:
        return np.nan, np.nan
    return (c_ks + 1) / (valid + 1), (c_ad + 1) / (valid + 1)
