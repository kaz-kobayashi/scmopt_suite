# GitHub リポジトリ設定手順

## 🔧 ローカルでの準備

### 1. Git初期化とファイル追加
```bash
cd /Users/kazuhiro/Documents/2509/scmopt_suite

# Git初期化（まだの場合）
git init

# ファイル追加
git add .

# 初期コミット
git commit -m "Initial commit: SCMOPT2 with authentication system

- Complete FastAPI backend with JWT authentication
- React frontend with optimization features
- Railway deployment configuration
- Docker containerization
- PostgreSQL and Redis support"
```

## 🌐 GitHub リポジトリ作成

### 2. GitHub上でリポジトリ作成
1. https://github.com にアクセス
2. "New repository" ボタンをクリック
3. Repository name: `scmopt_suite` 
4. Description: `Supply Chain Management Optimization Suite with Authentication`
5. Public または Private を選択
6. **README, .gitignore, license は追加しない**（既に作成済み）
7. "Create repository" ボタンをクリック

### 3. ローカルリポジトリとGitHubを接続
```bash
# GitHubリポジトリを追加（YOUR_USERNAMEを実際のユーザー名に変更）
git remote add origin https://github.com/YOUR_USERNAME/scmopt_suite.git

# mainブランチに変更（必要に応じて）
git branch -M main

# GitHubにプッシュ
git push -u origin main
```

## 🚀 Railway デプロイ手順

### 4. Railway連携
1. https://railway.app にアクセス
2. "New Project" をクリック
3. "Deploy from GitHub repo" を選択
4. `scmopt_suite` リポジトリを選択
5. Root directory: `backend` を指定

### 5. 環境変数設定
Railway Dashboard → Settings → Variables で以下を追加:
```
SECRET_KEY=ランダムな32文字の文字列
PYTHONPATH=/app
PORT=8000
```

### 6. データベース追加
- "Add Database" → "PostgreSQL"
- "Add Database" → "Redis"

### 7. デプロイ設定確認
Settings → Deploy で以下を確認:
```
Build Command: pip install -r requirements.txt
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## ✅ 確認事項

### デプロイ後のテスト
```bash
# ヘルスチェック
curl https://your-app.up.railway.app/health

# ユーザー登録テスト
curl -X POST https://your-app.up.railway.app/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"testpass","full_name":"Test User"}'
```

## 🔄 継続的デプロイ

GitHubにpushするたびに自動デプロイされます：
```bash
git add .
git commit -m "Update: new feature"
git push origin main
```

## 💰 料金
- **GitHub**: 無料（パブリックリポジトリ）
- **Railway**: 月500時間無料、その後$5/月