"""ArtemisFit — ajuste de datos a distribuciones de probabilidad (Equipo Artemis, Simulación 2026A)."""
import pandas as pd
import streamlit as st
from scipy import stats

from core import data as D, export as E, fitting as F, plots as P

st.set_page_config(page_title="ArtemisFit", page_icon="🏹", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:3.2rem;max-width:1280px}
.stApp{background:#F6F5FC}
section[data-testid="stSidebar"]{background:#EAE7FA}
span[data-baseweb="tag"]{background:#7C3AED!important;color:#fff!important}
.hero{background:linear-gradient(120deg,#1e1b4b 0%,#5b21b6 55%,#0e7490 100%);border-radius:22px;
  padding:28px 36px;color:#fff;margin-bottom:10px;box-shadow:0 12px 32px rgba(76,29,149,.28)}
.hero h1{margin:0;font-size:2.5rem;color:#fff;padding:0}
.hero p{margin:.35rem 0 0;color:#e0e7ff;font-size:1.05rem}
.sec{display:flex;align-items:center;gap:12px;margin:38px 0 14px}
.sec .num{background:linear-gradient(135deg,#7c3aed,#14b8a6);color:#fff;border-radius:50%;width:36px;
  height:36px;display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0}
.sec h2{margin:0;padding:0;font-size:1.5rem;color:#1f2340}
.kpi{background:#fff;border-radius:14px;padding:13px 16px;border-left:5px solid var(--c);
  box-shadow:0 3px 12px rgba(31,35,64,.08)}
.kpi .l{font-size:.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:.05em}
.kpi .v{font-size:1.4rem;font-weight:700;color:#1f2340;line-height:1.3}
.best{background:linear-gradient(135deg,#ede9fe,#ccfbf1);border-radius:20px;padding:24px 28px;
  border:1px solid #ddd6fe;box-shadow:0 6px 20px rgba(124,58,237,.12)}
.best .t{font-size:.8rem;color:#6d28d9;font-weight:800;text-transform:uppercase;letter-spacing:.08em}
.best .n{font-size:2.1rem;font-weight:800;color:#1f2340;line-height:1.2}
.best .p{font-family:monospace;color:#374151;margin:.2rem 0 .6rem}
.best .r{color:#1f2340;font-size:1rem}
.pill{display:inline-block;padding:3px 13px;border-radius:999px;font-size:.8rem;font-weight:600;margin:4px 6px 0 0}
.ok{background:#d1fae5;color:#065f46}.warn{background:#fef3c7;color:#92400e}
.ch{font-weight:700;padding:7px 15px;border-radius:10px;color:#fff;margin-bottom:6px;display:inline-block}
div[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;box-shadow:0 3px 12px rgba(31,35,64,.08)}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2{color:#5b21b6}
</style>""", unsafe_allow_html=True)

TR = {
    "es": dict(
        sub="Ajuste de datos a distribuciones de probabilidad para simulación",
        lang="Idioma / Language", src="📥 Origen de los datos", paste="Pegar / escribir", upload="Subir archivo",
        example="Ejemplo", hint="Un dato por línea, o separados por espacio, ; o ,", file="CSV / Excel",
        column="Columna", opts="⚙️ Opciones", shift="Permitir desplazamiento (loc)",
        dists="Distribuciones", boot="Réplicas bootstrap",
        start="👈 Pega tus datos, sube un archivo o elige **Ejemplo** en la barra lateral.",
        s1="Tus datos", s2="Mejor ajuste", s3="Explora y compara", s4="Llévalo a tu simulador",
        n="Datos", mean="Media", median="Mediana", sd="Desv. est.", skew="Asimetría", kurt="Curtosis",
        range="Rango", indep="Independencia (supuesto i.i.d.)", lag1="Autocorrelación lag-1",
        runs="Prueba de rachas, p =", ind_ok="Sin evidencia contra la independencia",
        ind_bad="Posible dependencia: el ajuste i.i.d. podría no ser válido",
        best="🏆 Mejor ajuste según AIC", chi_ok="χ²: no se rechaza el ajuste", chi_bad="χ²: revisar el ajuste",
        also="Mejor por ranking global:", rec="es la mejor opción según AIC/BIC",
        equiv="Prácticamente equivalentes (ΔAIC < 2):", full="Ver todas las métricas",
        alt_note="No hay expresión directa para {a}. Sustituto más cercano del ranking:",
        equiv_hint="Con ajustes equivalentes conviene elegir el más simple o el que tenga expresión directa en tu simulador.",
        table_note="K-S y χ² son referenciales (parámetros estimados de los datos). "
                   "Para p-valores corregidos use el bootstrap de la sección 3.",
        analyze="Distribución a analizar", compare="Comparar con", bootb="🎲 Calcular p-valores con bootstrap",
        corrected="p-valores corregidos (bootstrap)",
        no_expr="No hay expresión directa; use el código SimPy/SciPy.",
        sim="Simulador: muestra generada vs. datos", n_sim="Tamaño de la muestra simulada",
        dl_x="⬇️ Descargar Excel", dl_c="⬇️ Descargar ranking (CSV)",
        cols={"label": "Distribución", "params": "Parámetros", "dAIC": "ΔAIC", "chi2_p": "χ² p", "rank_mean": "Ranking global", "KS": "K-S", "AD": "A-D"}, none="Pocos resultados: revise los datos.",
    ),
    "en": dict(
        sub="Fitting data to probability distributions for simulation",
        lang="Idioma / Language", src="📥 Data source", paste="Paste / type", upload="Upload file",
        example="Example", hint="One value per line, or separated by space, ; or ,", file="CSV / Excel",
        column="Column", opts="⚙️ Options", shift="Allow shift (loc)",
        dists="Distributions", boot="Bootstrap replications",
        start="👈 Paste your data, upload a file or pick **Example** in the sidebar.",
        s1="Your data", s2="Best fit", s3="Explore and compare", s4="Take it to your simulator",
        n="Data", mean="Mean", median="Median", sd="Std. dev.", skew="Skewness", kurt="Kurtosis",
        range="Range", indep="Independence (i.i.d. assumption)", lag1="Lag-1 autocorrelation",
        runs="Runs test, p =", ind_ok="No evidence against independence",
        ind_bad="Possible dependence: an i.i.d. fit may not be valid",
        best="🏆 Best fit by AIC", chi_ok="χ²: fit not rejected", chi_bad="χ²: review the fit",
        also="Best overall rank:", rec="is the best option by AIC/BIC",
        equiv="Practically equivalent (ΔAIC < 2):", full="Show all metrics",
        alt_note="No direct expression for {a}. Closest substitute in the ranking:",
        equiv_hint="With equivalent fits, prefer the simplest one or the one with a direct expression in your simulator.",
        table_note="K-S and χ² are indicative (parameters estimated from the data). "
                   "Use the bootstrap in section 3 for corrected p-values.",
        analyze="Distribution to analyze", compare="Compare with", bootb="🎲 Compute bootstrap p-values",
        corrected="Corrected p-values (bootstrap)",
        no_expr="No direct expression; use the SimPy/SciPy code.",
        sim="Simulator: generated sample vs. data", n_sim="Simulated sample size",
        dl_x="⬇️ Download Excel", dl_c="⬇️ Download ranking (CSV)",
        cols={"label": "Distribution", "params": "Parameters", "dAIC": "ΔAIC", "chi2_p": "χ² p", "rank_mean": "Overall rank", "KS": "K-S", "AD": "A-D"}, none="Few results: check your data.",
    ),
}

lang = st.sidebar.radio(TR["es"]["lang"], ["es", "en"], horizontal=True)
t = TR[lang]


def sec(n, title):
    st.markdown(f'<div class="sec"><div class="num">{n}</div><h2>{title}</h2></div>', unsafe_allow_html=True)


def kpi(col, label, value, color="#7C3AED"):
    col.markdown(f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div>'
                 f'<div class="v">{value}</div></div>', unsafe_allow_html=True)


def fnum(v, d=4):
    return "—" if v != v else f"{v:.{d}g}"


st.markdown(f'<div class="hero"><h1>🏹 ArtemisFit</h1><p>{t["sub"]}</p></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner="Ajustando distribuciones…")
def run_fit(x, keys, shift):
    return F.fit_all(x, list(keys), shift)


# ------------------------------------------------------------ barra lateral
st.sidebar.header(t["src"])
src = st.sidebar.radio("src", [t["paste"], t["upload"], t["example"]], index=2, label_visibility="collapsed")
data = None
try:
    if src == t["paste"]:
        txt = st.sidebar.text_area(t["hint"], height=170)
        if txt.strip():
            data = D.parse_text(txt)
    elif src == t["upload"]:
        f = st.sidebar.file_uploader(t["file"], type=["csv", "txt", "xlsx", "xls"])
        if f:
            df = D.read_table(f)
            cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 8]
            data = D.clean_series(df[st.sidebar.selectbox(t["column"], cols)])
    else:
        data = stats.weibull_min.rvs(1.8, loc=2, scale=10, size=300, random_state=2026)
except ValueError as e:
    st.sidebar.error(str(e))

if data is None:
    st.info(t["start"])
    st.stop()

discrete = D.is_discrete(data)
st.sidebar.header(t["opts"])
shift = st.sidebar.checkbox(t["shift"], value=False)
all_keys = list(F.CONTINUOUS) + list(F.DISCRETE)
keys = st.sidebar.multiselect(t["dists"], all_keys, format_func=F.LABELS.get,
                              default=all_keys if discrete else list(F.CONTINUOUS))
n_boot = st.sidebar.slider(t["boot"], 50, 500, 200, 50)

results = run_fit(data, tuple(keys), shift)
if not results:
    st.error(t["none"])
    st.stop()
table = F.results_table(results)
by_key = {r.key: r for r in results}
best = results[0]

# ============================================================ 1. DATOS
sec(1, t["s1"])
d = D.describe(data)
cols = st.columns(6)
for c, (lab, val, color) in zip(cols, [
        (t["n"], f"{d['n']}", "#7C3AED"), (t["mean"], fnum(d["media"]), "#14B8A6"),
        (t["median"], fnum(d["mediana"]), "#F59E0B"), (t["sd"], fnum(d["desv. est."]), "#EC4899"),
        (t["skew"], fnum(d["asimetría"]), "#3B82F6"), (t["kurt"], fnum(d["curtosis (exceso)"]), "#10B981")]):
    kpi(c, lab, val, color)
st.write("")
c1, c2 = st.columns([3, 2])
c1.plotly_chart(P.histogram_pdf(data, [], lang), width="stretch")
with c2:
    z, p = D.runs_test(data)
    good = p > 0.05
    st.markdown(f"**{t['indep']}**")
    st.markdown(f'<span class="pill {"ok" if good else "warn"}">{t["ind_ok"] if good else t["ind_bad"]}</span>',
                unsafe_allow_html=True)
    st.write(f"{t['lag1']}: **{D.autocorr1(data):.3f}**")
    st.write(f"{t['runs']} **{p:.3f}**")
    st.write(f"{t['range']}: **[{fnum(d['min'])}, {fnum(d['max'])}]**")

# ============================================================ 2. MEJOR AJUSTE
sec(2, t["s2"])
chi_good = best.chi2_p == best.chi2_p and best.chi2_p > 0.05
table["dAIC"] = table["AIC"] - table["AIC"].min()
alt = table.sort_values("rank_mean").iloc[0]["label"]
also = f" {t['also']} <b>{alt}</b>." if alt != best.label else ""
equiv = [x.label for x in results[1:] if x.aic - best.aic < 2]
eq_html = (f'<div class="r" style="margin-top:8px">💡 {t["equiv"]} <b>{", ".join(equiv)}</b>. '
           f'{t["equiv_hint"]}</div>') if equiv else ""
st.markdown(f"""
<div class="best"><div class="t">{t["best"]}</div>
<div class="n">{best.label}</div><div class="p">{F.params_str(best.key, best.params)}</div>
<div class="r"><b>{best.label}</b> {t["rec"]} (AIC = {best.aic:.2f} · BIC = {best.bic:.2f}).{also}</div>{eq_html}
<span class="pill {"ok" if chi_good else "warn"}">{t["chi_ok"] if chi_good else t["chi_bad"]}
(p = {fnum(best.chi2_p, 3)})</span></div>""", unsafe_allow_html=True)
st.write("")
main_cols = ["label", "params", "AIC", "dAIC", "BIC", "KS", "AD", "chi2_p", "rank_mean"]


def render_table(df, height):
    fmt = {c: "{:.4f}" for c in df.select_dtypes("float").columns}
    st.dataframe(df.style.format(fmt, na_rep="—").apply(
        lambda r: ["background-color:#EDE9FE;font-weight:600" if r.name == 0 else "" for _ in r], axis=1),
        width="stretch", hide_index=True, height=height)


render_table(table[main_cols].rename(columns=t["cols"]), min(420, 38 * (len(table) + 1)))
with st.expander(t["full"]):
    render_table(table.drop(columns=["key"]).rename(columns=t["cols"]), 400)
st.caption(t["table_note"])

# ============================================================ 3. EXPLORA
sec(3, t["s3"])
a, b = st.columns([1, 2])
sel = a.selectbox(t["analyze"], [r.key for r in results], format_func=F.LABELS.get)
cmp_keys = b.multiselect(t["compare"], [r.key for r in results], default=[r.key for r in results[:3]],
                         format_func=F.LABELS.get)
r = by_key[sel]
plotted = [by_key[k] for k in dict.fromkeys([sel] + cmp_keys)]
k5 = st.columns(5)
kpi(k5[0], "AIC", fnum(r.aic, 6), "#7C3AED")
kpi(k5[1], "BIC", fnum(r.bic, 6), "#14B8A6")
kpi(k5[2], "K-S (p ref.)", f"{fnum(r.ks, 3)} ({fnum(r.ks_p, 2)})", "#F59E0B")
kpi(k5[3], "Anderson-Darling", fnum(r.ad, 4), "#EC4899")
kpi(k5[4], "χ² (p)", f"{fnum(r.chi2, 4)} ({fnum(r.chi2_p, 2)})", "#3B82F6")
st.write("")
if not r.discrete and st.button(t["bootb"]):
    with st.spinner("Monte Carlo…"):
        pk, pa = F.bootstrap_pvalues(data, r, shift, n_boot)
    st.markdown(f'**{t["corrected"]}** — <span class="pill {"ok" if pk > .05 else "warn"}">K-S p = {pk:.3f}</span>'
                f'<span class="pill {"ok" if pa > .05 else "warn"}">A-D p = {pa:.3f}</span>',
                unsafe_allow_html=True)
g1, g2 = st.columns(2)
g1.plotly_chart(P.histogram_pdf(data, plotted, lang), width="stretch")
g2.plotly_chart(P.cdf_plot(data, plotted, lang), width="stretch")
g3, g4 = st.columns(2)
g3.plotly_chart(P.qq_plot(data, r, lang), width="stretch")
g4.plotly_chart(P.pp_plot(data, r, lang), width="stretch")

# ============================================================ 4. SIMULADOR
sec(4, t["s4"])
st.markdown(f"**{r.label}** — `{F.params_str(r.key, r.params)}`")
x1, x2, x3 = st.columns(3)
for col, name, color, fn in [(x1, "Arena", "#7C3AED", E.arena_expr), (x2, "Simio", "#14B8A6", E.simio_expr)]:
    col.markdown(f'<span class="ch" style="background:{color}">{name}</span>', unsafe_allow_html=True)
    expr, sub = E.with_fallback(results, r, fn)
    if sub:
        col.caption(t["alt_note"].format(a=r.label) + f" **{sub}**")
    col.code(expr or t["no_expr"], language=None, wrap_lines=True)
x3.markdown('<span class="ch" style="background:#F59E0B">SimPy / SciPy</span>', unsafe_allow_html=True)
x3.code(E.simpy_code(r), language="python", wrap_lines=True)

st.markdown(f"##### {t['sim']}")
s1, s2 = st.columns([1, 2])
n_sim = s1.number_input(t["n_sim"], 100, 100000, len(data), 100)
samp = r.frozen().rvs(size=int(n_sim), random_state=0)
cmp_ = pd.DataFrame({"datos" if lang == "es" else "data": D.describe(data),
                     "simulada" if lang == "es" else "simulated": D.describe(samp.astype(float))}).drop(index="n")
s2.dataframe(cmp_.style.format("{:.4f}"), width="stretch")
if not r.discrete:
    s1.caption(f"K-S 2 muestras: p = {stats.ks_2samp(data, samp).pvalue:.3f}")
d1, d2 = st.columns(2)
d1.download_button(t["dl_x"], E.excel_bytes(data, results), "artemisfit_resultados.xlsx", width="stretch")
d2.download_button(t["dl_c"], table.to_csv(index=False).encode(), "artemisfit_ranking.csv", width="stretch")
st.markdown("<br><center style='color:#9ca3af'>🏹 ArtemisFit · Equipo Artemis · Simulación 2026A</center>",
            unsafe_allow_html=True)
