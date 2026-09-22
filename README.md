# Backend — Monitor de Cartera

API en Flask que calcula las métricas de la cartera (pestaña "Panel de Análisis").
Reutiliza el motor de cálculo original de tu `app.py`, sin nada de Streamlit.

## Probarlo local

```bash
pip install -r requirements.txt
python app.py
```

Luego, en otra terminal:

```bash
curl -X POST http://localhost:5000/api/dashboard \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["BRK-B","GOOGL","IBM","JPM","MU","NU","NVDA","SPY"],
    "pesos": {"BRK-B":15.08,"GOOGL":16.73,"IBM":8.32,"JPM":20.31,"MU":11.35,"NU":4.69,"NVDA":17.38,"SPY":6.14},
    "fecha_inicio": "2020-10-09",
    "fecha_fin": "2026-09-09",
    "benchmark": "SPY",
    "rf": 4.2
  }'
```

## Deployar gratis en Render

1. Subí esta carpeta (`backend/`) a un repo de GitHub.
2. En [render.com](https://render.com) → **New +** → **Web Service** → conectá el repo.
3. Configuración:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free
4. Render te da una URL tipo `https://monitor-cartera-api.onrender.com`. Esa URL es la que el frontend en Netlify va a llamar.

**Nota sobre el plan free de Render:** el servicio "se duerme" tras 15 min sin uso y tarda ~30-50 seg en despertar en la primera consulta del día. Es normal y no cuesta nada; si más adelante querés que responda siempre al instante, existe el plan pago desde USD 7/mes.

## Siguientes endpoints a agregar (pestañas 2 a 5)

- `/api/screener/lynch` y `/api/screener/buffett` — motor de screening fundamental (ya está como función pura en tu `app.py`: `evaluar_criterios`, `obtener_fundamentales_screener`).
- `/api/sec/<ticker>` — datos de SEC EDGAR y sector.
- `/api/pe-ratio/<ticker>` — serie histórica de PE Ratio vs EPS TTM.

Se agregan como nuevas rutas en `app.py`, reutilizando la misma lógica que ya tenés escrita — el patrón es siempre: función pura en `finance_engine.py` → ruta Flask que la llama y devuelve JSON.
