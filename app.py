"""ArtemisFit — ajuste de datos a distribuciones de probabilidad (Equipo Artemis, Simulación 2026A)."""
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from core import data as D, export as E, fitting as F, plots as P

st.set_page_config(page_title="ArtemisFit", page_icon="🏹", layout="wide")

TR = {
    "es": dict(
        subtitle="Ajuste de datos a distribuciones de probabilidad para simulación",
        source="Origen de los datos", paste="Pegar / escribir", upload="Subir archivo", example="Ejemplo",
        paste_hint="Un dato por línea, o separados por espacio, ; o ,", file="CSV / Excel", column="Columna",
        shift="Permitir desplazamiento (loc)", dists="Distribuciones", boot="Réplicas bootstrap",
        start="Pega, sube o genera datos en la barra lateral para comenzar.",
        t_data="Datos", t_rank="Ranking", t_plots="Gráficos", t_export="Exportar y simular",
        desc="Estadísticos descriptivos", indep="Independencia (supuesto i.i.d.)",
        lag1="Autocorrelación lag-1", runs="Prueba de rachas (p-valor)",
        indep_ok="No hay evidencia contra la independencia.",
        indep_bad="Posible dependencia: el ajuste de una distribución i.i.d. podría no ser válido.",
        rank_note="Los p-valores KS/χ² son referenciales (parámetros estimados de los datos). "
                  "Use el bootstrap en la pestaña Gráficos para p-valores corregidos.",
        best_aic="Mejor por AIC", best_rank="Mejor ranking global", pick="Distribución",
        pick_many="Distribuciones a comparar", run_boot="Calcular p-valores con bootstrap",
        corrected="p-valores corregidos (bootstrap)", rec="Recomendación",
        n_sim="Tamaño de muestra simulada", sim="Simulador: muestra generada vs. datos",
        dl_xlsx="Descargar Excel", dl_csv="Descargar ranking (CSV)", expr="Expresiones para simuladores",
        no_expr="No hay expresión directa; use el código SimPy/SciPy.",
        warn_few="Pocos resultados: revise que los datos sean válidos.",
    ),
    "en": dict(
        subtitle="Fitting data to probability distributions for simulation",
        source="Data source", paste="Paste / type", upload="Upload file", example="Example",
        paste_hint="One value per line, or separated by space, ; or ,", file="CSV / Excel", column="Column",
        shift="Allow shift (loc)", dists="Distributions", boot="Bootstrap replications",
        start="Paste, upload or generate data in the sidebar to begin.",
        t_data="Data", t_rank="Ranking", t_plots="Charts", t_export="Export & simulate",
        desc="Descriptive statistics", indep="Independence (i.i.d. assumption)",
        lag1="Lag-1 autocorrelation", runs="Runs test (p-value)",
        indep_ok="No evidence against independence.",
        indep_bad="Possible dependence: fitting an i.i.d. distribution may not be valid.",
        rank_note="KS/χ² p-values are indicative (parameters estimated from the data). "
                  "Use the bootstrap in the Charts tab for corrected p-values.",
        best_aic="Best by AIC", best_rank="Best overall rank", pick="Distribution",
        pick_many="Distributions to compare", run_boot="Compute bootstrap p-values",
        corrected="Corrected p-values (bootstrap)", rec="Recommendation",
        n_sim="Simulated sample size", sim="Simulator: generated sample vs. data",
        dl_xlsx="Download Excel", dl_csv="Download ranking (CSV)", expr="Simulator expressions",
        no_expr="No direct expression; use the SimPy/SciPy code.",
        warn_few="Few results: check that the data are valid.",
    ),
}
lang = st.sidebar.radio("Idioma / Language", ["es", "en"], horizontal=True)
t = lambda k: TR[lang][k]

st.title("🏹 ArtemisFit")
st.caption(t("subtitle"))


@st.cache_data(show_spinner="Ajustando distribuciones…")
def run_fit(x, keys, shift):
    return F.fit_all(x, list(keys), shift)


# ------------------------------------------------------------ entrada
src = st.sidebar.radio(t("source"), [t("paste"), t("upload"), t("example")])
data = None
try:
    if src == t("paste"):
        txt = st.sidebar.text_area(t("paste_hint"), height=180)
        if txt.strip():
            data = D.parse_text(txt)
    elif src == t("upload"):
        f = st.sidebar.file_uploader(t("file"), type=["csv", "txt", "xlsx", "xls"])
        if f:
            df = D.read_table(f)
            cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 8]
            col = st.sidebar.selectbox(t("column"), cols)
            data = D.clean_series(df[col])
    else:
        data = stats.weibull_min.rvs(1.8, loc=2, scale=10, size=300, random_state=2026)
except ValueError as e:
    st.sidebar.error(str(e))

if data is None:
    st.info(t("start"))
    st.stop()

discrete = D.is_discrete(data)
shift = st.sidebar.checkbox(t("shift"), value=False)
all_keys = list(F.CONTINUOUS) + list(F.DISCRETE)
keys = st.sidebar.multiselect(t("dists"), all_keys, default=all_keys if discrete else list(F.CONTINUOUS),
                              format_func=F.LABELS.get)
n_boot = st.sidebar.slider(t("boot"), 50, 500, 200, 50)

results = run_fit(data, tuple(keys), shift)
if len(results) < 1:
    st.error(t("warn_few"))
    st.stop()
table = F.results_table(results)
by_key = {r.key: r for r in results}

tab_data, tab_rank, tab_plots, tab_exp = st.tabs([t("t_data"), t("t_rank"), t("t_plots"), t("t_export")])

# ------------------------------------------------------------ datos
with tab_data:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader(t("desc"))
        st.dataframe(pd.Series(D.describe(data), name="valor").to_frame().style.format("{:.5g}"))
        st.subheader(t("indep"))
        z, p = D.runs_test(data)
        st.write(f"{t('lag1')}: **{D.autocorr1(data):.3f}** · {t('runs')}: **{p:.3f}**")
        (st.success if p > 0.05 else st.warning)(t("indep_ok") if p > 0.05 else t("indep_bad"))
    with c2:
        st.plotly_chart(P.histogram_pdf(data, [], lang), width="stretch")

# ------------------------------------------------------------ ranking
with tab_rank:
    best_aic = table.sort_values("AIC").iloc[0]
    best_rank = table.sort_values("rank_mean").iloc[0]
    m1, m2 = st.columns(2)
    m1.metric(t("best_aic"), best_aic["label"])
    m2.metric(t("best_rank"), best_rank["label"])
    show = table.drop(columns=["key"]).rename(columns={"label": "Distribución" if lang == "es" else "Distribution",
                                                       "params": "Parámetros" if lang == "es" else "Parameters"})
    st.dataframe(show.style.format({c: "{:.4f}" for c in show.select_dtypes("float").columns}, na_rep="—"),
                 width="stretch", hide_index=True)
    st.caption(t("rank_note"))

# ------------------------------------------------------------ gráficos
with tab_plots:
    top = [r.key for r in results[:3]]
    many = st.multiselect(t("pick_many"), [r.key for r in results], default=top, format_func=F.LABELS.get)
    sel_res = [by_key[k] for k in many]
    if sel_res:
        c1, c2 = st.columns(2)
        c1.plotly_chart(P.histogram_pdf(data, sel_res, lang), width="stretch")
        c2.plotly_chart(P.cdf_plot(data, sel_res, lang), width="stretch")
    one = st.selectbox(t("pick"), [r.key for r in results], format_func=F.LABELS.get, key="plot_one")
    res = by_key[one]
    c1, c2 = st.columns(2)
    c1.plotly_chart(P.qq_plot(data, res), width="stretch")
    c2.plotly_chart(P.pp_plot(data, res), width="stretch")
    if not res.discrete and st.button(t("run_boot")):
        with st.spinner("Monte Carlo…"):
            pk, pa = F.bootstrap_pvalues(data, res, shift, n_boot)
        st.write(f"**{t('corrected')}** — KS: {pk:.3f} · Anderson-Darling: {pa:.3f}")

# ------------------------------------------------------------ exportar
with tab_exp:
    pick = st.selectbox(t("pick"), [r.key for r in results], format_func=F.LABELS.get, key="exp_pick")
    r = by_key[pick]
    st.markdown(f"**{r.label}** — {F.params_str(r.key, r.params)}")
    verdict = "no se rechaza" if (r.chi2_p == r.chi2_p and r.chi2_p > 0.05) else "revisar"
    st.info(f"{t('rec')}: {r.label} (AIC={r.aic:.2f}, BIC={r.bic:.2f}); χ² p={r.chi2_p:.3f} → {verdict}.")
    st.subheader(t("expr"))
    a, s_ = E.arena_expr(r), E.simio_expr(r)
    st.markdown("**Arena**"); st.code(a or t("no_expr"))
    st.markdown("**Simio**"); st.code(s_ or t("no_expr"))
    st.markdown("**SimPy / SciPy**"); st.code(E.simpy_code(r), language="python")

    st.subheader(t("sim"))
    n_sim = st.number_input(t("n_sim"), 100, 100000, len(data), 100)
    samp = r.frozen().rvs(size=int(n_sim), random_state=0)
    cmp_ = pd.DataFrame({"datos": D.describe(data), "simulada": D.describe(samp.astype(float))}).drop(index="n")
    st.dataframe(cmp_.style.format("{:.4f}"))
    if not r.discrete:
        st.caption(f"KS 2 muestras: p = {stats.ks_2samp(data, samp).pvalue:.3f}")

    d1, d2 = st.columns(2)
    d1.download_button(t("dl_xlsx"), E.excel_bytes(data, results), "artemisfit_resultados.xlsx")
    d2.download_button(t("dl_csv"), table.to_csv(index=False).encode(), "artemisfit_ranking.csv")
