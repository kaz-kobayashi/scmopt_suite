# Railway 最新デプロイ手順

## 🔄 Railway CLI 最新コマンド

### 環境変数設定
```bash
# 正しいコマンド形式
railway variables --set SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
railway variables --set PYTHONPATH=/app

# 確認
railway variables
```

### プロジェクト管理
```bash
# 初期化
railway init

# デプロイ
railway up

# ログ確認
railway logs

# ダッシュボード開く
railway open
```

## 🌐 Web UI推奨手順

### 1. GitHub連携（最も簡単）
1. https://railway.app/new でプロジェクト作成
2. "Deploy from GitHub repo" 選択
3. リポジトリ選択 → `backend` フォルダ指定

### 2. 設定項目
```
Root Directory: backend
Build Command: pip install -r requirements.txt
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT

Environment Variables:
- SECRET_KEY: [ランダム32文字]
- PYTHONPATH: /app
```

### 3. データベース追加
- Add Database → PostgreSQL
- Add Database → Redis

## ⚡ 即座デプロイ

### GitHubからの場合
```bash
# 1. GitHubにpush
git add .
git commit -m "Deploy to Railway"
git push origin main

# 2. Railway Dashboard
# - New Project
# - Deploy from GitHub repo
# - 自動デプロイ開始
```

### CLI使用の場合
```bash
# 1. プロジェクト初期化
railway init

# 2. 環境変数設定
railway variables --set SECRET_KEY=$(openssl rand -hex 32)
railway variables --set PYTHONPATH=/app

# 3. デプロイ
railway up --detach
```

## 🔍 トラブルシューティング

### CLI エラーの場合
```bash
# CLIアップデート
npm update -g @railway/cli

# または brew経由
brew upgrade railway
```

### Web UI使用推奨
- より直感的
- エラーが少ない
- リアルタイムログ表示