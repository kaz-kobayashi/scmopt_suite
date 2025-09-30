from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import mimetypes
from app.routes import analytics, inventory, routing, lnd, lotsize, scrm, snd, rm, shift, jobshop, templates, auth
# PyVRP routes temporarily disabled due to installation issues
# from app.routes import pyvrp_routes, async_vrp_routes, websocket_routes, advanced_vrp_routes, batch_vrp_routes
from app.routers import realtime, websocket, data_import_export
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

# Include routers
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(inventory.router, prefix="/api/inventory", tags=["inventory"]) 
app.include_router(routing.router, prefix="/api/routing", tags=["routing"])
app.include_router(lnd.router, prefix="/api/lnd", tags=["logistics-network-design"])
app.include_router(lotsize.router, prefix="/api/lotsize", tags=["lot-size-optimization"])
app.include_router(scrm.router, prefix="/api/scrm", tags=["supply-chain-risk-management"])
app.include_router(snd.router, prefix="/api/snd", tags=["service-network-design"])
app.include_router(rm.router, prefix="/api/rm", tags=["revenue-management"])
app.include_router(shift.router, prefix="/api/shift", tags=["shift-optimization"])
app.include_router(jobshop.router, prefix="/api/jobshop", tags=["job-shop-scheduling"])
app.include_router(templates.router, prefix="/api/templates", tags=["schedule-templates"])
# PyVRP routes temporarily disabled due to installation issues
# app.include_router(pyvrp_routes.router, prefix="/api/pyvrp", tags=["advanced-vehicle-routing"])
# app.include_router(async_vrp_routes.router, prefix="/api/vrp/v1", tags=["async-vehicle-routing"])
# app.include_router(advanced_vrp_routes.router, prefix="/api/vrp/v1/advanced", tags=["advanced-optimization"])
# app.include_router(batch_vrp_routes.router, prefix="/api/vrp/v1", tags=["batch-processing"])
# app.include_router(websocket_routes.router, prefix="/api/vrp/v1", tags=["websocket-notifications"])
app.include_router(realtime.router, prefix="/api/realtime", tags=["realtime-scheduling"])
app.include_router(websocket.router, tags=["websocket"])
app.include_router(data_import_export.router, prefix="/api", tags=["data-import-export"])
app.include_router(auth.router, prefix="/api", tags=["authentication"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/debug/static")
async def debug_static_files():
    """デバッグ用：静的ファイルの状況を確認"""
    static_dir = "static"
    result = {
        "static_dir_exists": os.path.exists(static_dir),
        "current_working_dir": os.getcwd(),
        "static_dir_path": os.path.abspath(static_dir) if os.path.exists(static_dir) else None,
    }
    
    if os.path.exists(static_dir):
        try:
            files = os.listdir(static_dir)
            result["files_in_static"] = files[:10]  # 最初の10ファイルのみ
            result["index_html_exists"] = "index.html" in files
            
            if "index.html" in files:
                index_path = os.path.join(static_dir, "index.html")
                result["index_html_size"] = os.path.getsize(index_path)
        except Exception as e:
            result["error_reading_static"] = str(e)
    
    return result

# 静的ファイル（React）を配信
static_dir = "static"

@app.get("/")
async def serve_react_app():
    """Serve the React app's index.html"""
    if os.path.exists(static_dir):
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        else:
            return {"error": "index.html not found", "static_dir_exists": True}
    else:
        return {"message": "SCMOPT2 API Server", "version": "2.0.0", "status": "running", "static_dir_exists": False}

# Catch-all route for SPA and static files
@app.get("/{path:path}")
async def serve_static_or_spa(path: str):
    """Serve static files or SPA routes"""
    # Don't catch API routes, health check, or debug routes
    if path.startswith("api/") or path == "health" or path.startswith("debug/"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not Found")
    
    if os.path.exists(static_dir):
        # Check if it's a static file (CSS, JS, images, etc.)
        file_path = os.path.join(static_dir, path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            # Determine media type based on file extension
            media_type, _ = mimetypes.guess_type(file_path)
            return FileResponse(file_path, media_type=media_type)
        
        # For all SPA routes, serve index.html
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path, media_type="text/html")
        else:
            return {"error": "index.html not found for SPA route", "path": path}
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Static files not found")