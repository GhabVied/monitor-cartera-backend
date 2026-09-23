# ============================================================
#  app.py — API del Monitor de Cartera (Flask)
#  Expone el motor de cálculo (finance_engine.py) como endpoints
#  HTTP para que el frontend (alojado en Netlify) los consuma.
#
#  Correr local:      python app.py
#  Deployar (Render):  ver README.md en esta misma carpeta.
# ============================================================

import math

from flask import Flask, jsonify, request
from flask_cors import CORS

from finance_engine import analizar_cartera, evaluar_screener, test_estres

app = Flask(__name__)

# Reemplazá "*" por el dominio final de tu sitio en Netlify una vez
# que lo tengas (ej. "https://monitor-de-cartera.netlify.app") para
# que solo tu propio frontend pueda llamar a esta API.
CORS(app, resources={r"/api/*": {"origins": "*"}})


def _limpiar_nan(obj):
    """Reemplaza NaN/Infinity (no válidos en JSON estricto) por None."""
    if isinstance(obj, dict):
        return {k: _limpiar_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_limpiar_nan(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/dashboard")
def dashboard():
    """
    Body JSON esperado:
    {
      "tickers": ["BRK-B", "GOOGL", "IBM", "JPM", "MU", "NU", "NVDA", "SPY"],
      "pesos": {"BRK-B": 15.08, "GOOGL": 16.73, ...},
      "fecha_inicio": "2020-10-09",
      "fecha_fin": "2026-09-09",
      "benchmark": "SPY",
      "rf": 4.2
    }
    """
    data = request.get_json(force=True, silent=True) or {}

    try:
        resultado = analizar_cartera(
            tickers=data.get("tickers", []),
            pesos_raw=data.get("pesos", {}),
            fecha_inicio=data["fecha_inicio"],
            fecha_fin=data["fecha_fin"],
            benchmark_ticker=data.get("benchmark", "SPY"),
            rf_pct=float(data.get("rf", 0)),
            pesos_minimos=data.get("pesos_minimos"),
            pesos_maximos=data.get("pesos_maximos"),
            permitir_corto=bool(data.get("permitir_corto", False)),
        )
        return jsonify(_limpiar_nan(resultado))
    except KeyError as e:
        return jsonify({"error": f"Falta el campo {e}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001 — devolvemos el error tal cual para debug
        return jsonify({"error": f"Error interno: {e}"}), 500


@app.post("/api/stress-test")
def stress_test():
    """
    Body JSON esperado:
    {
      "tickers": [...],
      "pesos": {...},
      "beta": 1.15   // la beta de la cartera, calculada por /api/dashboard
    }
    Tarda más que /api/dashboard porque descarga historial de hasta 35+ años.
    """
    data = request.get_json(force=True, silent=True) or {}
    try:
        resultado = test_estres(
            tickers=data.get("tickers", []),
            pesos_raw=data.get("pesos", {}),
            beta_cartera=float(data.get("beta", 1.0)),
        )
        return jsonify(_limpiar_nan({"eventos": resultado}))
    except KeyError as e:
        return jsonify({"error": f"Falta el campo {e}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Error interno: {e}"}), 500


@app.post("/api/screener")
def screener():
    """
    Body JSON esperado:
    {
      "tickers": [...],
      "pesos": {...},
      "estrategia": "lynch"   // o "buffett"
    }
    Consulta datos fundamentales de Yahoo Finance para cada ticker (en paralelo,
    con caché de 24hs), así que la primera vez puede tardar 10-30 segundos.
    """
    data = request.get_json(force=True, silent=True) or {}
    try:
        resultado = evaluar_screener(
            tickers=data.get("tickers", []),
            pesos_raw=data.get("pesos", {}),
            estrategia=data.get("estrategia", "lynch"),
        )
        return jsonify(_limpiar_nan(resultado))
    except KeyError as e:
        return jsonify({"error": f"Falta el campo {e}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Error interno: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
