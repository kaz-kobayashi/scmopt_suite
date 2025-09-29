from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routes import auth
from app.database import engine, Base
import os

app = FastAPI(
    title="SCMOPT2 API",
    description="Supply Chain Management Optimization Suite",
    version="2.0.0"
)

# Create database tables
Base.metadata.create_all(bind=engine)

# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth router only
app.include_router(auth.router, prefix="/api", tags=["authentication"])

@app.get("/")
async def root():
    return {"message": "SCMOPT2 API Server", "version": "2.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# 静的ファイル（React）を配信
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
    
    # SPAのルーティング対応
    @app.get("/{path:path}")
    async def serve_spa(path: str):
        file_path = f"static/{path}"
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse("static/index.html")