# Railway簡単デプロイ手順

## 🚀 最短デプロイ方法

### 1. GitHubにpush
```bash
# まだの場合はGitリポジトリ初期化
git init
git add .
git commit -m "Initial commit for Railway deployment"

# GitHubにpush (リポジトリ作成済みの場合)
git remote add origin https://github.com/yourusername/scmopt_suite.git
git push -u origin main
```

### 2. Railway Dashboard
1. https://railway.app にアクセス
2. GitHubでログイン
3. "New Project" クリック
4. "Deploy from GitHub repo" 選択
5. `scmopt_suite` リポジトリ選択
6. `backend` フォルダ選択

### 3. 環境変数設定
Settings → Variables で以下を追加:
```
SECRET_KEY=your-32-character-secret-key-here
PYTHONPATH=/app
PORT=8000
```

### 4. データベース追加
- "Add Database" → "PostgreSQL" 
- "Add Database" → "Redis"

### 5. デプロイ設定
Settings → Deploy で:
```
Root Directory: backend
Build Command: pip install -r requirements.txt  
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## 📊 Railway料金
- **無料枠**: 月500時間 ($5相当)
- **有料**: $5/月でunlimited usage
- **データベース**: PostgreSQL/Redis込み

## 🔗 デプロイ後URL
```
https://your-project-name.up.railway.app
```

## 🧪 テスト用エンドポイント
```bash
# ヘルスチェック
curl https://your-project-name.up.railway.app/health

# ユーザー登録
curl -X POST https://your-project-name.up.railway.app/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"testpass","full_name":"Test User"}'
```