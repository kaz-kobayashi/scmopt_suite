#!/bin/sh

# PORT環境変数が設定されていない場合のデフォルト値
PORT=${PORT:-8000}

echo "Starting server on port $PORT..."

# 起動
exec uvicorn app.main_minimal:app --host 0.0.0.0 --port $PORT --log-level debug