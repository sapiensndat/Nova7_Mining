web: gunicorn --bind 0.0.0.0:$PORT gold:app
worker: celery -A worker.celery_app worker --loglevel=inf