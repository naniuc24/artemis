# 🏹 ArtemisFit
Ajuste de datos a distribuciones de probabilidad — Equipo Artemis, Simulación 2026A.

## Ejecutar local
```bash
pip install -r requirements.txt
streamlit run app.py
pytest            # validación contra SciPy
```

## Funciones
- Datos: pegar, CSV/Excel, ejemplo. Descriptivos, autocorrelación y prueba de rachas (i.i.d.).
- 21 distribuciones continuas + 3 discretas, MLE (SciPy), desplazamiento opcional.
- Ranking: AIC, BIC, Kolmogorov-Smirnov, Anderson-Darling, χ².
- Bootstrap paramétrico para p-valores corregidos de KS y A-D.
- Gráficos: histograma+PDF, CDF, Q-Q, P-P.
- Exportación: Arena, Simio, SimPy/SciPy, Excel, CSV. Simulador de muestras. ES/EN.

## Despliegue gratuito
Suba el repo a GitHub → https://share.streamlit.io → New app → `app.py`.
