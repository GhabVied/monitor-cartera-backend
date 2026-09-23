# ============================================================
#  finance_engine.py
#  Motor de cálculo financiero — extraído de app.py (Streamlit)
#  Sin ninguna dependencia de Streamlit: son funciones puras que
#  reciben datos y devuelven datos. Las usa app_api.py (Flask)
#  para exponerlas como endpoints HTTP, pero se pueden probar o
#  reutilizar de forma completamente independiente.
# ============================================================

import datetime as dt
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

# En servidores como Render, la carpeta donde yfinance guarda su caché
# interno (zonas horarias, etc.) puede no tener permiso de escritura,
# lo que produce el error "database is locked". /tmp sí es escribible.
try:
    yf.set_tz_cache_location("/tmp/yfinance_cache")
except Exception:
    pass

DIAS_BURSATILES = 252


# ------------------------------------------------------------------
# DATOS DE MERCADO
# ------------------------------------------------------------------

def descargar_precios(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Descarga precios de cierre ajustados desde Yahoo Finance.

    Devuelve un DataFrame de precios (fecha x ticker), ya limpio de
    columnas completamente vacías y con forward-fill de huecos menores.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        precios = raw["Close"]
    else:
        precios = raw[["Close"]]
        precios.columns = tickers

    precios = precios.dropna(axis=1, how="all").ffill().dropna(how="all")
    return precios


def calcular_retornos(precios: pd.DataFrame) -> pd.DataFrame:
    return precios.pct_change().dropna(how="all")


def descargar_historial_extendido(tickers: list[str], desde: str = "1989-01-01") -> pd.DataFrame:
    """Descarga el historial más largo posible de precios (para el test de estrés)."""
    raw = yf.download(tickers, start=desde, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        precios = raw["Close"]
    else:
        precios = raw[["Close"]]
        precios.columns = tickers
    return precios.dropna(axis=1, how="all")


# ------------------------------------------------------------------
# BASE DE DATOS DE CRISIS HISTÓRICAS (para el test de estrés)
# ------------------------------------------------------------------

CRISIS_HISTORICAS = [
    {"evento": "Guerra del Golfo / Shock del petróleo", "categoria": "Geopolítica / energía",
     "peak": "1990-07-16", "trough": "1990-10-11", "recovery": "1991-02-13"},
    {"evento": "Crisis Financiera Global (2008)", "categoria": "Crisis financiera / crediticia",
     "peak": "2007-10-09", "trough": "2009-03-09", "recovery": "2013-03-28"},
    {"evento": "Crisis de Deuda EE.UU. / Europa", "categoria": "Crisis soberana / macro",
     "peak": "2011-04-29", "trough": "2011-10-03", "recovery": "2012-02-28"},
    {"evento": "Selloff Q4 2018", "categoria": "Suba de tasas / guerra comercial",
     "peak": "2018-09-20", "trough": "2018-12-24", "recovery": "2019-04-23"},
    {"evento": "Crash COVID-19", "categoria": "Pandemia / shock exógeno",
     "peak": "2020-02-19", "trough": "2020-03-23", "recovery": "2020-08-18"},
    {"evento": "Bear Market 2022", "categoria": "Inflación / suba de tasas",
     "peak": "2022-01-03", "trough": "2022-10-12", "recovery": "2024-01-19"},
]


def analizar_estres_historico(
    precios_extendidos: pd.DataFrame,
    precio_indice: pd.Series,
    pesos: pd.Series,
    beta_cartera: float,
) -> list[dict]:
    """Para cada crisis histórica, calcula la caída real del índice y estima/calcula
    la caída de la cartera actual. Devuelve una lista de dicts lista para JSON."""
    primeras_fechas = precios_extendidos.apply(lambda s: s.first_valid_index())
    filas = []

    for crisis in CRISIS_HISTORICAS:
        peak = pd.Timestamp(crisis["peak"])
        trough = pd.Timestamp(crisis["trough"])
        recovery = pd.Timestamp(crisis["recovery"])

        indice_hasta_peak = precio_indice.loc[:peak].dropna()
        ventana_indice = precio_indice.loc[peak:trough].dropna()
        dd_indice = None
        if len(indice_hasta_peak) and len(ventana_indice):
            valor_pico = indice_hasta_peak.iloc[-1]
            dd_indice = float(ventana_indice.min() / valor_pico - 1)

        cobertura_completa = all(
            (primeras_fechas.get(t) is not None) and (primeras_fechas[t] <= peak)
            for t in pesos.index
        )

        dd_cartera_real = None
        if cobertura_completa:
            sub_precios = precios_extendidos[pesos.index].loc[peak:trough].dropna()
            if len(sub_precios) > 1:
                sub_retornos = sub_precios.pct_change().dropna()
                ret_cartera = (sub_retornos * pesos).sum(axis=1)
                dd_cartera_real = float((1 + ret_cartera).cumprod().min() - 1)

        dd_cartera_estimada = beta_cartera * dd_indice if dd_indice is not None else None

        filas.append({
            "evento": crisis["evento"],
            "categoria": crisis["categoria"],
            "periodo": f"{peak.date()} → {trough.date()}",
            "caida_indice_pct": round(dd_indice * 100, 2) if dd_indice is not None else None,
            "caida_cartera_estimada_pct": round(dd_cartera_estimada * 100, 2) if dd_cartera_estimada is not None else None,
            "caida_cartera_real_pct": round(dd_cartera_real * 100, 2) if dd_cartera_real is not None else None,
            "cobertura_completa": cobertura_completa,
            "dias_recuperacion": (recovery - peak).days,
        })

    return filas


# ------------------------------------------------------------------
# MÉTRICAS DE RIESGO Y RETORNO
# ------------------------------------------------------------------

def retorno_anualizado(retornos: pd.Series) -> float:
    """CAGR geométrico anualizado a partir de retornos diarios."""
    retornos = retornos.dropna()
    if len(retornos) == 0:
        return np.nan
    total = (1 + retornos).prod()
    anios = len(retornos) / DIAS_BURSATILES
    if anios <= 0 or total <= 0:
        return np.nan
    return total ** (1 / anios) - 1


def volatilidad_anualizada(retornos: pd.Series) -> float:
    return retornos.std() * np.sqrt(DIAS_BURSATILES)


def ratio_sharpe(retornos: pd.Series, rf: float) -> float:
    vol = volatilidad_anualizada(retornos)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (retorno_anualizado(retornos) - rf) / vol


def ratio_sortino(retornos: pd.Series, rf: float) -> float:
    downside = retornos[retornos < 0].std() * np.sqrt(DIAS_BURSATILES)
    if not downside or np.isnan(downside):
        return np.nan
    return (retorno_anualizado(retornos) - rf) / downside


def max_drawdown(retornos: pd.Series) -> float:
    acumulado = (1 + retornos).cumprod()
    pico = acumulado.cummax()
    drawdown = acumulado / pico - 1
    return drawdown.min()


def serie_drawdown(retornos: pd.Series) -> pd.Series:
    acumulado = (1 + retornos).cumprod()
    pico = acumulado.cummax()
    return acumulado / pico - 1


def calmar_ratio(retornos: pd.Series) -> float:
    mdd = abs(max_drawdown(retornos))
    if mdd == 0:
        return np.nan
    return retorno_anualizado(retornos) / mdd


def var_historico(retornos: pd.Series, nivel: float = 0.05) -> float:
    return float(np.percentile(retornos.dropna(), nivel * 100))


def cvar_historico(retornos: pd.Series, nivel: float = 0.05) -> float:
    var = var_historico(retornos, nivel)
    cola = retornos[retornos <= var]
    return float(cola.mean()) if len(cola) else np.nan


def beta_vs_benchmark(retornos: pd.Series, retornos_bm: pd.Series) -> float:
    df = pd.concat([retornos, retornos_bm], axis=1).dropna()
    if len(df) < 10:
        return np.nan
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1]
    var = np.var(df.iloc[:, 1])
    return cov / var if var != 0 else np.nan


def metricas_cartera(retornos: pd.Series, retornos_bm: pd.Series, rf: float) -> dict:
    """Calcula el set completo de métricas institucionales para una serie de retornos."""
    return {
        "cagr": retorno_anualizado(retornos),
        "vol_anual": volatilidad_anualizada(retornos),
        "sharpe": ratio_sharpe(retornos, rf),
        "sortino": ratio_sortino(retornos, rf),
        "calmar": calmar_ratio(retornos),
        "max_drawdown": max_drawdown(retornos),
        "var_95": var_historico(retornos),
        "cvar_95": cvar_historico(retornos),
        "beta": beta_vs_benchmark(retornos, retornos_bm),
    }


def descomponer_riesgo(pesos: pd.Series, retornos: pd.DataFrame, retornos_cartera: pd.Series) -> tuple[list[dict], float]:
    """Descompone la volatilidad total de la cartera en el aporte de cada activo."""
    activos = list(pesos.index)
    cov = retornos[activos].cov().values * DIAS_BURSATILES
    w = pesos.reindex(activos).values
    vol_cartera = float(np.sqrt(w @ cov @ w))

    mcr = np.zeros_like(w) if vol_cartera == 0 else (cov @ w) / vol_cartera
    ccr = w * mcr
    pct_riesgo = ccr / vol_cartera if vol_cartera != 0 else np.zeros_like(w)
    vol_individual = np.sqrt(np.diag(cov))
    correlaciones = [retornos[t].corr(retornos_cartera) for t in activos]

    filas = []
    for i, t in enumerate(activos):
        filas.append({
            "activo": t,
            "peso_pct": w[i] * 100,
            "vol_anual_pct": vol_individual[i] * 100,
            "correlacion": correlaciones[i],
            "riesgo_pct": pct_riesgo[i] * 100,
            "delta_vs_peso": pct_riesgo[i] * 100 - w[i] * 100,
        })
    filas.sort(key=lambda f: f["riesgo_pct"], reverse=True)
    return filas, vol_cartera


def matriz_correlacion(retornos: pd.DataFrame) -> dict:
    if retornos.shape[1] < 2:
        return {"tickers": list(retornos.columns), "matriz": []}
    corr = retornos.corr().round(4)
    return {"tickers": list(corr.columns), "matriz": corr.values.tolist()}


# ------------------------------------------------------------------
# OPTIMIZACIÓN DE CARTERA (TEORÍA MODERNA DE PORTAFOLIO — MARKOWITZ)
# ------------------------------------------------------------------

def _retorno_cartera_w(w: np.ndarray, mu: np.ndarray) -> float:
    return float(np.dot(w, mu))


def _volatilidad_cartera_w(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(np.dot(w.T, np.dot(cov, w))))


def preparar_inputs_optimizacion(retornos: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    activos = list(retornos.columns)
    mu = retornos.mean().values * DIAS_BURSATILES
    cov = retornos.cov().values * DIAS_BURSATILES
    return mu, cov, activos


def optimizar_cartera(
    mu: np.ndarray,
    cov: np.ndarray,
    objetivo: str,
    rf: float = 0.0,
    retorno_objetivo: float | None = None,
    permitir_corto: bool = False,
    pesos_minimos: np.ndarray | None = None,
    pesos_maximos: np.ndarray | None = None,
) -> np.ndarray | None:
    """objetivo: 'min_vol' | 'max_sharpe' | 'retorno_objetivo'."""
    import scipy.optimize as sco

    n = len(mu)
    piso_default = -1.0 if permitir_corto else 0.0
    if pesos_minimos is None:
        pesos_minimos = np.zeros(n)
    if pesos_maximos is None:
        pesos_maximos = np.ones(n)

    pisos = np.maximum(piso_default, pesos_minimos)
    techos = np.minimum(1.0, pesos_maximos)
    techos = np.maximum(techos, pisos)  # nunca un techo menor que su propio piso

    if pisos.sum() > 1.0 + 1e-9 or techos.sum() < 1.0 - 1e-9:
        return None

    bounds = tuple((float(pisos[i]), float(techos[i])) for i in range(n))
    remanente = max(0.0, 1.0 - pisos.sum())
    w0 = pisos + remanente / n
    w0 = np.minimum(w0, techos)

    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if objetivo == "min_vol":
        fun = lambda w: _volatilidad_cartera_w(w, cov)
    elif objetivo == "max_sharpe":
        fun = lambda w: -(_retorno_cartera_w(w, mu) - rf) / _volatilidad_cartera_w(w, cov)
    elif objetivo == "retorno_objetivo":
        fun = lambda w: _volatilidad_cartera_w(w, cov)
        restricciones.append({"type": "eq", "fun": lambda w: _retorno_cartera_w(w, mu) - retorno_objetivo})
    else:
        raise ValueError(f"Objetivo desconocido: {objetivo}")

    resultado = sco.minimize(
        fun, w0, method="SLSQP", bounds=bounds, constraints=restricciones,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    return resultado.x if resultado.success else None


def calcular_frontera_eficiente(
    mu: np.ndarray, cov: np.ndarray, n_puntos: int = 30,
    permitir_corto: bool = False, pesos_minimos: np.ndarray | None = None,
    pesos_maximos: np.ndarray | None = None,
) -> list[dict]:
    ret_min, ret_max = mu.min(), mu.max()
    objetivos = np.linspace(ret_min, ret_max, n_puntos)
    puntos = []
    for r_obj in objetivos:
        w = optimizar_cartera(
            mu, cov, "retorno_objetivo", retorno_objetivo=r_obj,
            permitir_corto=permitir_corto, pesos_minimos=pesos_minimos, pesos_maximos=pesos_maximos,
        )
        if w is not None:
            puntos.append({"retorno": r_obj, "volatilidad": _volatilidad_cartera_w(w, cov)})
    return puntos


def optimizacion_completa(
    retornos: pd.DataFrame, pesos_usuario: pd.Series, rf: float,
    permitir_corto: bool = False, pesos_minimos_dict: dict | None = None,
    pesos_maximos_dict: dict | None = None,
) -> dict | None:
    """Arma toda la sección de optimización: cartera min-vol, max-sharpe, frontera
    y la comparación con la cartera del usuario. None si no se pudo resolver."""
    if retornos.shape[1] < 2:
        return None

    mu, cov, activos_opt = preparar_inputs_optimizacion(retornos)
    pesos_minimos_dict = pesos_minimos_dict or {}
    pesos_maximos_dict = pesos_maximos_dict or {}
    vector_minimos = np.array([pesos_minimos_dict.get(t, 0.0) / 100 for t in activos_opt])
    vector_maximos = np.array([pesos_maximos_dict.get(t, 100.0) / 100 for t in activos_opt])

    if vector_minimos.sum() > 1.0 + 1e-9:
        return {"error": "La suma de los pesos mínimos supera el 100%."}
    if vector_maximos.sum() < 1.0 - 1e-9:
        return {"error": "La suma de los pesos máximos es menor al 100%: no hay forma de completar la cartera."}
    if np.any(vector_minimos > vector_maximos + 1e-9):
        return {"error": "Hay al menos un activo cuyo mínimo es mayor que su máximo."}

    w_min_vol = optimizar_cartera(mu, cov, "min_vol", permitir_corto=permitir_corto, pesos_minimos=vector_minimos, pesos_maximos=vector_maximos)
    w_max_sharpe = optimizar_cartera(mu, cov, "max_sharpe", rf=rf, permitir_corto=permitir_corto, pesos_minimos=vector_minimos, pesos_maximos=vector_maximos)
    frontera = calcular_frontera_eficiente(mu, cov, permitir_corto=permitir_corto, pesos_minimos=vector_minimos, pesos_maximos=vector_maximos)

    if w_min_vol is None or w_max_sharpe is None:
        return {"error": "No se pudo resolver la optimización con los datos y restricciones actuales."}

    pesos_u = np.array([pesos_usuario.get(t, 0.0) for t in activos_opt])

    def resumen(w):
        ret = _retorno_cartera_w(w, mu)
        vol = _volatilidad_cartera_w(w, cov)
        sharpe = (ret - rf) / vol if vol > 0 else float("nan")
        return {"retorno_pct": ret * 100, "vol_pct": vol * 100, "sharpe": sharpe}

    return {
        "activos": activos_opt,
        "individuales": [
            {"activo": t, "vol_pct": float(np.sqrt(cov[i, i])) * 100, "retorno_pct": float(mu[i]) * 100}
            for i, t in enumerate(activos_opt)
        ],
        "frontera": [{"retorno_pct": p["retorno"] * 100, "vol_pct": p["volatilidad"] * 100} for p in frontera],
        "tu_cartera": resumen(pesos_u),
        "min_volatilidad": resumen(w_min_vol),
        "max_sharpe": resumen(w_max_sharpe),
        "pesos_comparados": [
            {
                "activo": t,
                "tu_cartera_pct": pesos_u[i] * 100,
                "min_vol_pct": w_min_vol[i] * 100,
                "max_sharpe_pct": w_max_sharpe[i] * 100,
            }
            for i, t in enumerate(activos_opt)
        ],
    }


# ------------------------------------------------------------------
# ARMADO COMPLETO DEL PANEL (equivale a lo que hacía la pestaña
# "Panel de Análisis" del Streamlit original, sin nada de UI)
# ------------------------------------------------------------------

def analizar_cartera(
    tickers: list[str],
    pesos_raw: dict[str, float],
    fecha_inicio: str,
    fecha_fin: str,
    benchmark_ticker: str,
    rf_pct: float,
    pesos_minimos: dict[str, float] | None = None,
    pesos_maximos: dict[str, float] | None = None,
    permitir_corto: bool = False,
) -> dict:
    """Punto de entrada único para el endpoint /api/dashboard.

    tickers: lista de tickers de la cartera (sin el benchmark).
    pesos_raw: {ticker: peso en %}, no hace falta que sumen 100 (se normalizan).
    fecha_inicio / fecha_fin: 'YYYY-MM-DD'.
    rf_pct: tasa libre de riesgo anual en % (ej. 4.2, no 0.042).
    pesos_minimos / pesos_maximos: {ticker: % en 0-100}, usados solo para la
    optimización de cartera (no afectan tu cartera actual, que usa pesos_raw).
    permitir_corto: si True, la optimización puede asignar pesos negativos.

    Devuelve un dict listo para serializar a JSON con las métricas,
    la curva de equity (cartera vs benchmark, base 100) y el detalle
    por activo — todo lo que necesita el frontend para pintar el
    Panel de Análisis.
    """
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise ValueError("Ingresá al menos un ticker.")

    suma_pesos = sum(pesos_raw.values())
    if suma_pesos <= 0:
        raise ValueError("La suma de los pesos debe ser mayor a 0.")
    pesos = {t: p / suma_pesos for t, p in pesos_raw.items()}

    inicio = dt.date.fromisoformat(fecha_inicio)
    fin = dt.date.fromisoformat(fecha_fin)
    if inicio >= fin:
        raise ValueError("La fecha de inicio debe ser anterior a la fecha de fin.")

    rf = rf_pct / 100

    todos_los_tickers = list(dict.fromkeys(tickers + [benchmark_ticker]))
    precios = descargar_precios(todos_los_tickers, inicio, fin)
    if precios.empty:
        raise ValueError("No se pudieron descargar datos para los tickers indicados.")

    tickers_validos = [t for t in tickers if t in precios.columns]
    faltantes = [t for t in tickers if t not in precios.columns]
    if not tickers_validos:
        raise ValueError("Ninguno de los tickers ingresados tiene datos disponibles.")
    if benchmark_ticker not in precios.columns:
        raise ValueError(f"No se pudo descargar el benchmark '{benchmark_ticker}'.")

    pesos_validos = pd.Series({t: pesos[t] for t in tickers_validos})
    pesos_validos = pesos_validos / pesos_validos.sum()

    retornos = calcular_retornos(precios[tickers_validos])
    retornos_bm = calcular_retornos(precios[[benchmark_ticker]])[benchmark_ticker]
    retornos, retornos_bm = retornos.align(retornos_bm, join="inner", axis=0)

    retornos_cartera = (retornos * pesos_validos).sum(axis=1)
    retornos_cartera.name = "Cartera"

    met = metricas_cartera(retornos_cartera, retornos_bm, rf)
    met_bm = metricas_cartera(retornos_bm, retornos_bm, rf)

    # Detalle por activo
    activos = []
    for t in tickers_validos:
        r = retornos[t].dropna()
        activos.append({
            "ticker": t,
            "peso_pct": pesos_validos[t] * 100,
            "retorno_anual_pct": retorno_anualizado(r) * 100,
            "vol_anual_pct": volatilidad_anualizada(r) * 100,
            "sharpe": ratio_sharpe(r, rf),
            "max_drawdown_pct": max_drawdown(r) * 100,
            "beta": beta_vs_benchmark(r, retornos_bm),
        })
    activos.sort(key=lambda a: a["peso_pct"], reverse=True)

    # Curva de equity, base 100, para el gráfico de evolución
    equity_cartera = (1 + retornos_cartera).cumprod() * 100
    equity_bm = (1 + retornos_bm).cumprod() * 100
    fechas = [d.strftime("%Y-%m-%d") for d in equity_cartera.index]

    # Drawdown de la cartera
    dd_cartera = serie_drawdown(retornos_cartera) * 100

    # Matriz de correlación entre activos
    correlacion = matriz_correlacion(retornos)

    # Descomposición de riesgo por activo
    riesgo_por_activo, vol_cartera_desc = descomponer_riesgo(pesos_validos, retornos, retornos_cartera)

    # Optimización de cartera (frontera eficiente, min-vol, max-sharpe)
    optimizacion = optimizacion_completa(
        retornos, pesos_validos, rf,
        permitir_corto=permitir_corto,
        pesos_minimos_dict=pesos_minimos,
        pesos_maximos_dict=pesos_maximos,
    )

    return {
        "fechas": fechas,
        "equity_cartera": [round(v, 4) for v in equity_cartera.tolist()],
        "equity_benchmark": [round(v, 4) for v in equity_bm.tolist()],
        "drawdown_cartera": [round(v, 4) for v in dd_cartera.tolist()],
        "metricas_cartera": met,
        "metricas_benchmark": met_bm,
        "activos": activos,
        "correlacion": correlacion,
        "riesgo_por_activo": riesgo_por_activo,
        "vol_cartera_descomp_pct": vol_cartera_desc * 100,
        "optimizacion": optimizacion,
        "faltantes": faltantes,
        "benchmark": benchmark_ticker,
    }


def test_estres(
    tickers: list[str],
    pesos_raw: dict[str, float],
    beta_cartera: float,
) -> list[dict]:
    """Descarga el historial más largo posible y calcula las caídas de la cartera
    actual durante las crisis históricas más relevantes. Es una llamada aparte
    (más lenta) porque descarga hasta 35+ años de historial."""
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    suma = sum(pesos_raw.values())
    if suma <= 0:
        raise ValueError("La suma de los pesos debe ser mayor a 0.")
    pesos = pd.Series({t: pesos_raw[t] / suma for t in tickers if t in pesos_raw})

    precios_ext = descargar_historial_extendido(tickers + ["^GSPC"])
    if precios_ext.empty or "^GSPC" not in precios_ext.columns:
        raise ValueError("No se pudo descargar el historial extendido del S&P 500 (^GSPC).")

    indice = precios_ext["^GSPC"]
    tickers_disponibles = [t for t in pesos.index if t in precios_ext.columns]
    pesos = pesos.reindex(tickers_disponibles)
    pesos = pesos / pesos.sum()

    return analizar_estres_historico(precios_ext.drop(columns=["^GSPC"]), indice, pesos, beta_cartera)


# ============================================================
# SCREENING FUNDAMENTAL — Peter Lynch (GARP) y Buffett/Munger
# (Pestaña 2 y 3 de la app original)
# ============================================================

UMBRAL_MID_CAP = 2_000_000_000  # USD. Por debajo de esto se considera small cap.

# Cada criterio: (nombre, clave_en_fundamentales, texto_umbral, condicion, formateador)
CRITERIOS_LYNCH = [
    ("P/E", "pe", "< 25", lambda v: v < 25, lambda v: f"{v:.1f}"),
    ("Fwd P/E", "fwd_pe", "< 15", lambda v: v < 15, lambda v: f"{v:.1f}"),
    ("D/E", "de", "< 0.4", lambda v: v < 0.4, lambda v: f"{v:.2f}"),
    ("EPS YoY", "eps_yoy", "> 15%", lambda v: v > 0.15, lambda v: f"{v*100:.1f}%"),
    ("PEG", "peg", "< 2", lambda v: v < 2, lambda v: f"{v:.2f}"),
    ("Mid+", "market_cap", "≥ Mid Cap", lambda v: v >= UMBRAL_MID_CAP, lambda v: f"${v/1e9:.1f}B"),
]

CRITERIOS_BUFFETT = [
    ("ROE", "roe", "> 15%", lambda v: v > 0.15, lambda v: f"{v*100:.1f}%"),
    ("D/E", "de", "< 0.5", lambda v: v < 0.5, lambda v: f"{v:.2f}"),
    ("Gross Margin", "gross_margin", "> 30%", lambda v: v > 0.30, lambda v: f"{v*100:.1f}%"),
    ("P/E", "pe", "< 25", lambda v: v < 25, lambda v: f"{v:.1f}"),
    ("Profitable", "profit_margin", "> 0%", lambda v: v > 0, lambda v: f"{v*100:.1f}%"),
    ("Mid+", "market_cap", "≥ Mid Cap", lambda v: v >= UMBRAL_MID_CAP, lambda v: f"${v/1e9:.1f}B"),
]

ESTRATEGIAS_SCREENER = {"lynch": CRITERIOS_LYNCH, "buffett": CRITERIOS_BUFFETT}

# Caché simple en memoria (24hs), igual que el st.cache_data del original.
# Como Render free es un solo proceso, un dict alcanza; se pierde al reiniciar.
_CACHE_FUNDAMENTALES: dict[tuple, tuple] = {}
_CACHE_TTL_SEGUNDOS = 60 * 60 * 24


# Caché simple en memoria (24hs), igual que el st.cache_data del original.
# Como Render free es un solo proceso, un dict alcanza; se pierde al reiniciar.
_CACHE_FUNDAMENTALES: dict[tuple, tuple] = {}
_CACHE_TTL_SEGUNDOS = 60 * 60 * 24
_CRUMB_LOCK = threading.Lock()
_CRUMB_LISTO = False


def _asegurar_sesion_yahoo():
    """Pide la credencial de sesión ("crumb") de Yahoo Finance una sola vez,
    de forma secuencial. Si varios tickers la piden al mismo tiempo (en
    paralelo) se pisan entre sí y casi todas las consultas fallan con
    'Invalid Crumb' — por eso esto se hace ANTES de lanzar los hilos."""
    global _CRUMB_LISTO
    if _CRUMB_LISTO:
        return
    with _CRUMB_LOCK:
        if _CRUMB_LISTO:
            return
        try:
            yf.Ticker("AAPL").info  # dispara el handshake cookie+crumb una vez
        except Exception:
            pass
        _CRUMB_LISTO = True


def _fundamentales_de_un_ticker(t: str, intentos: int = 2) -> dict:
    ultimo_error = None
    for intento in range(intentos):
        try:
            info = yf.Ticker(t).info
            if not info or len(info) < 3:
                raise ValueError("Respuesta vacía de Yahoo Finance")
            de = info.get("debtToEquity")
            de = de / 100 if de is not None else None
            eps_yoy = info.get("earningsGrowth")
            if eps_yoy is None:
                eps_yoy = info.get("earningsQuarterlyGrowth")
            peg = info.get("pegRatio") or info.get("trailingPegRatio")
            return {
                "pe": info.get("trailingPE"),
                "fwd_pe": info.get("forwardPE"),
                "de": de,
                "eps_yoy": eps_yoy,
                "peg": peg,
                "market_cap": info.get("marketCap"),
                "roe": info.get("returnOnEquity"),
                "gross_margin": info.get("grossMargins"),
                "profit_margin": info.get("profitMargins"),
                "nombre": info.get("shortName") or t,
            }
        except Exception as e:
            ultimo_error = e
            time.sleep(0.8 * (intento + 1))
    return {
        "pe": None, "fwd_pe": None, "de": None, "eps_yoy": None, "peg": None,
        "market_cap": None, "roe": None, "gross_margin": None, "profit_margin": None,
        "nombre": t, "_error": str(ultimo_error) if ultimo_error else None,
    }


def obtener_fundamentales_screener(tickers: list[str]) -> dict:
    """Trae de Yahoo Finance los datos fundamentales usados por cualquiera de
    las estrategias de screening. Consulta cada ticker en paralelo (son
    llamadas independientes) y cachea el resultado combinado 24hs."""
    clave_cache = tuple(sorted(tickers))
    ahora = dt.datetime.utcnow().timestamp()
    if clave_cache in _CACHE_FUNDAMENTALES:
        guardado_en, datos_cacheados = _CACHE_FUNDAMENTALES[clave_cache]
        if ahora - guardado_en < _CACHE_TTL_SEGUNDOS:
            return datos_cacheados

    _asegurar_sesion_yahoo()

    datos = {}
    with ThreadPoolExecutor(max_workers=min(4, len(tickers) or 1)) as ex:
        futuros = {ex.submit(_fundamentales_de_un_ticker, t): t for t in tickers}
        for fut in as_completed(futuros):
            t = futuros[fut]
            datos[t] = fut.result()

    _CACHE_FUNDAMENTALES[clave_cache] = (ahora, datos)
    return datos


def evaluar_screener(tickers: list[str], pesos_raw: dict[str, float], estrategia: str) -> dict:
    """Punto de entrada único para /api/screener. estrategia: 'lynch' | 'buffett'."""
    if estrategia not in ESTRATEGIAS_SCREENER:
        raise ValueError(f"Estrategia desconocida: {estrategia}")
    criterios = ESTRATEGIAS_SCREENER[estrategia]

    tickers = [t.strip().upper() for t in tickers if t.strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise ValueError("Ingresá al menos un ticker.")

    suma = sum(pesos_raw.values()) or 1
    pesos_pct = {t: (pesos_raw.get(t, 0) / suma) * 100 for t in tickers}

    fundamentales = obtener_fundamentales_screener(tickers)

    filas = []
    for t in tickers:
        d = fundamentales.get(t, {})
        cumplidos, evaluables = 0, 0
        criterios_fila = []
        for nombre_col, clave, umbral_txt, condicion, formateador in criterios:
            valor = d.get(clave)
            if valor is None:
                ok = None
                valor_fmt = "N/D"
            else:
                try:
                    ok = bool(condicion(valor))
                    valor_fmt = formateador(valor)
                    evaluables += 1
                    cumplidos += int(ok)
                except Exception:
                    ok = None
                    valor_fmt = "N/D"
            criterios_fila.append({
                "nombre": nombre_col, "umbral": umbral_txt, "valor": valor_fmt, "ok": ok,
            })
        filas.append({
            "activo": t,
            "nombre": d.get("nombre", t),
            "peso_pct": pesos_pct.get(t, 0),
            "cumplidos": cumplidos,
            "evaluables": evaluables,
            "criterios": criterios_fila,
        })

    filas.sort(key=lambda f: (f["cumplidos"]/f["evaluables"] if f["evaluables"] else 0), reverse=True)

    cumple_mayoria = [f for f in filas if f["evaluables"] and f["cumplidos"] >= f["evaluables"] * 0.66]
    peso_cumple = sum(f["peso_pct"] for f in cumple_mayoria)
    promedio_cumplidos = sum(f["cumplidos"] for f in filas) / len(filas) if filas else 0

    return {
        "estrategia": estrategia,
        "filas": filas,
        "resumen": {
            "n_cumple_mayoria": len(cumple_mayoria),
            "n_total": len(filas),
            "peso_cumple_pct": peso_cumple,
            "promedio_cumplidos": promedio_cumplidos,
            "total_criterios": len(criterios),
        },
    }
