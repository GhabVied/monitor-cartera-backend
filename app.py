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

from finance_engine import analizar_cartera

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
        )
        return jsonify(_limpiar_nan(resultado))
    except KeyError as e:
        return jsonify({"error": f"Falta el campo {e}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001 — devolvemos el error tal cual para debug
        return jsonify({"error": f"Error interno: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
