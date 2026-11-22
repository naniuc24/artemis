"""Gráficos de diagnóstico con Plotly (paleta Artemis)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .fitting import FitResult

PALETTE = ["#14B8A6", "#F59E0B", "#EC4899", "#3B82F6", "#10B981", "#EF4444", "#8B5CF6"]
VIOLET = "#7C3AED"
BAR = "rgba(124,58,237,0.28)"


def _style(fig, title, h=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1F2340")),
        template="plotly_white", colorway=PALETTE, height=h,
        margin=dict(l=40, r=20, t=55, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.75)",
        legend=dict(orientation="h", y=-0.2), font=dict(family="sans-serif", color="#374151"),
    )
    return fig


def _nbins(x):
    n = len(x)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr > 0:
        h = 2 * iqr * n ** (-1 / 3)                      # Freedman-Diaconis
        return int(np.clip(np.ceil((x.max() - x.min()) / h), 5, 60))
    return int(np.ceil(np.log2(n) + 1))                  # Sturges


def histogram_pdf(x, results: list[FitResult], lang="es"):
    fig = go.Figure()
    disc = any(r.discrete for r in results) or (not results and np.all(np.mod(x, 1) == 0)
                                                 and len(np.unique(x)) < 40)
    if disc:
        v, c = np.unique(x, return_counts=True)
        fig.add_bar(x=v, y=c / len(x), name="Datos" if lang == "es" else "Data",
                    marker_color=BAR, marker_line_color=VIOLET, marker_line_width=1)
        grid = np.arange(int(x.min()), int(x.max()) + 1)
        for r in results:
            if r.discrete:
                fig.add_scatter(x=grid, y=r.frozen().pmf(grid), mode="lines+markers", name=r.label)
    else:
        fig.add_histogram(x=x, nbinsx=_nbins(x), histnorm="probability density",
                          name="Datos" if lang == "es" else "Data", marker_color=BAR,
                          marker_line_color=VIOLET, marker_line_width=1)
    grid = np.linspace(x.min(), x.max(), 400)
    for r in results:
        if not r.discrete:
            fig.add_scatter(x=grid, y=r.frozen().pdf(grid), mode="lines", name=r.label,
                            line=dict(width=3))
    fig.update_layout(xaxis_title="x", yaxis_title="f(x)", bargap=0.03)
    if not results:
        title = "Histograma de los datos" if lang == "es" else "Data histogram"
    else:
        title = "Histograma y densidades ajustadas" if lang == "es" else "Histogram and fitted densities"
    return _style(fig, title)


def cdf_plot(x, results: list[FitResult], lang="es"):
    xs = np.sort(x)
    ecdf = np.arange(1, len(xs) + 1) / len(xs)
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ecdf, mode="lines", line_shape="hv", name="ECDF",
                    line=dict(color="#1F2340", width=2))
    for r in results:
        fig.add_scatter(x=xs, y=r.frozen().cdf(xs), mode="lines", name=r.label, line=dict(width=3))
    fig.update_layout(xaxis_title="x", yaxis_title="F(x)")
    return _style(fig, "CDF empírica vs. teórica" if lang == "es" else "Empirical vs. fitted CDF")


def qq_plot(x, res: FitResult, lang="es"):
    xs = np.sort(x)
    p = (np.arange(1, len(xs) + 1) - 0.5) / len(xs)
    q = res.frozen().ppf(p)
    lim = [min(xs.min(), np.nanmin(q)), max(xs.max(), np.nanmax(q))]
    fig = go.Figure()
    fig.add_scatter(x=q, y=xs, mode="markers", marker=dict(color=VIOLET, size=6, opacity=0.75))
    fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="#F59E0B", width=2))
    fig.update_layout(showlegend=False,
                      xaxis_title="Cuantiles teóricos" if lang == "es" else "Theoretical quantiles",
                      yaxis_title="Cuantiles de los datos" if lang == "es" else "Sample quantiles")
    return _style(fig, f"Q-Q plot — {res.label}")


def pp_plot(x, res: FitResult, lang="es"):
    xs = np.sort(x)
    p = (np.arange(1, len(xs) + 1) - 0.5) / len(xs)
    fig = go.Figure()
    fig.add_scatter(x=p, y=res.frozen().cdf(xs), mode="markers",
                    marker=dict(color="#14B8A6", size=6, opacity=0.75))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#F59E0B", width=2))
    fig.update_layout(showlegend=False,
                      xaxis_title="Probabilidad empírica" if lang == "es" else "Empirical probability",
                      yaxis_title="Probabilidad teórica" if lang == "es" else "Theoretical probability")
    return _style(fig, f"P-P plot — {res.label}")
