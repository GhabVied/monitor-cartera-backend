# ============================================================
#  finance_engine.py
#  Motor de cálculo financiero — extraído de app.py (Streamlit)
#  Sin ninguna dependencia de Streamlit: son funciones puras que
#  reciben datos y devuelven datos. Las usa app_api.py (Flask)
#  para exponerlas como endpoints HTTP, pero se pueden probar o
#  reutilizar de forma completamente independiente.
# ============================================================

import datetime as dt

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
) -> dict:
    """Punto de entrada único para el endpoint /api/dashboard.

    tickers: lista de tickers de la cartera (sin el benchmark).
    pesos_raw: {ticker: peso en %}, no hace falta que sumen 100 (se normalizan).
    fecha_inicio / fecha_fin: 'YYYY-MM-DD'.
    rf_pct: tasa libre de riesgo anual en % (ej. 4.2, no 0.042).

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

    return {
        "fechas": fechas,
        "equity_cartera": [round(v, 4) for v in equity_cartera.tolist()],
        "equity_benchmark": [round(v, 4) for v in equity_bm.tolist()],
        "metricas_cartera": met,
        "metricas_benchmark": met_bm,
        "activos": activos,
        "faltantes": faltantes,
        "benchmark": benchmark_ticker,
    }
