# gunicorn lee este archivo automáticamente al arrancar (no hace falta
# tocar el "Start Command" en Render). Subimos el timeout por defecto (30s)
# porque el test de estrés y el screener fundamental pueden tardar más,
# sobre todo la primera vez que se consulta un ticker (sin caché).
timeout = 120
workers = 1
threads = 4
