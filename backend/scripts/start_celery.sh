#!/bin/bash

# Start Celery worker and beat scheduler for local development

echo "Starting Redis server..."
redis-server --daemonize yes

echo "Starting Celery worker..."
celery -A celery_app worker --loglevel=info --concurrency=4 &

echo "Starting Celery beat scheduler..."
celery -A celery_app beat --loglevel=info &

echo "Starting Flower monitoring..."
celery -A celery_app flower --port=5555 &

echo "All Celery services started!"
echo "Flower monitoring available at: http://localhost:5555"
echo "Redis running on: redis://localhost:6379/0"

# Keep script running
wait