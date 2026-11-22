"""Gráficos de diagnóstico con Plotly."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .fitting import FitResult


def _nbins(x):
    n = len(x)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr > 0:
        h = 2 * iqr * n ** (-1 / 3)                      # Freedman-Diaconis
        return int(np.clip(np.ceil((x.max() - x.min()) / h), 5, 60))
    return int(np.ceil(np.log2(n) + 1))                  # Sturges


def histogram_pdf(x, results: list[FitResult], lang="es"):
    fig = go.Figure()
    disc = any(r.discrete for r in results)
    if disc:
        v, c = np.unique(x, return_counts=True)
        fig.add_bar(x=v, y=c / len(x), name="Datos", opacity=0.5)
        grid = np.arange(int(x.min()), int(x.max()) + 1)
        for r in results:
            if r.discrete:
                fig.add_scatter(x=grid, y=r.frozen().pmf(grid), mode="lines+markers", name=r.label)
    else:
        fig.add_histogram(x=x, nbinsx=_nbins(x), histnorm="probability density",
                          name="Datos", opacity=0.5)
    if not disc or any(not r.discrete for r in results):
        grid = np.linspace(x.min(), x.max(), 400)
        for r in results:
            if not r.discrete:
                fig.add_scatter(x=grid, y=r.frozen().pdf(grid), mode="lines", name=r.label)
    fig.update_layout(title="Histograma y densidades ajustadas" if lang == "es"
                      else "Histogram and fitted densities",
                      xaxis_title="x", yaxis_title="f(x)", bargap=0.02, legend_orientation="h")
    return fig


def cdf_plot(x, results: list[FitResult], lang="es"):
    xs = np.sort(x)
    ecdf = np.arange(1, len(xs) + 1) / len(xs)
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ecdf, mode="lines", line_shape="hv", name="ECDF",
                    line=dict(color="black"))
    for r in results:
        fig.add_scatter(x=xs, y=r.frozen().cdf(xs), mode="lines", name=r.label)
    fig.update_layout(title="CDF empírica vs. teórica" if lang == "es" else "Empirical vs. fitted CDF",
                      xaxis_title="x", yaxis_title="F(x)", legend_orientation="h")
    return fig


def qq_plot(x, res: FitResult):
    xs = np.sort(x)
    p = (np.arange(1, len(xs) + 1) - 0.5) / len(xs)
    q = res.frozen().ppf(p)
    lim = [min(xs.min(), np.nanmin(q)), max(xs.max(), np.nanmax(q))]
    fig = go.Figure()
    fig.add_scatter(x=q, y=xs, mode="markers", name="Q-Q")
    fig.add_scatter(x=lim, y=lim, mode="lines", name="y = x", line=dict(dash="dash"))
    fig.update_layout(title=f"Q-Q plot — {res.label}", xaxis_title="Cuantiles teóricos",
                      yaxis_title="Cuantiles de los datos", showlegend=False)
    return fig


def pp_plot(x, res: FitResult):
    xs = np.sort(x)
    p = (np.arange(1, len(xs) + 1) - 0.5) / len(xs)
    fig = go.Figure()
    fig.add_scatter(x=p, y=res.frozen().cdf(xs), mode="markers", name="P-P")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"))
    fig.update_layout(title=f"P-P plot — {res.label}", xaxis_title="Probabilidad empírica",
                      yaxis_title="Probabilidad teórica", showlegend=False)
    return fig
