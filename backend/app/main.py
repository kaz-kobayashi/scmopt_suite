from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routes import analytics, inventory, routing, lnd, lotsize, scrm, snd, rm, shift, jobshop, templates, pyvrp_routes, async_vrp_routes, websocket_routes, advanced_vrp_routes, batch_vrp_routes, auth
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
app.include_router(pyvrp_routes.router, prefix="/api/pyvrp", tags=["advanced-vehicle-routing"])
app.include_router(async_vrp_routes.router, prefix="/api/vrp/v1", tags=["async-vehicle-routing"])
app.include_router(advanced_vrp_routes.router, prefix="/api/vrp/v1/advanced", tags=["advanced-optimization"])
app.include_router(batch_vrp_routes.router, prefix="/api/vrp/v1", tags=["batch-processing"])
app.include_router(websocket_routes.router, prefix="/api/vrp/v1", tags=["websocket-notifications"])
app.include_router(realtime.router, prefix="/api/realtime", tags=["realtime-scheduling"])
app.include_router(websocket.router, tags=["websocket"])
app.include_router(data_import_export.router, prefix="/api", tags=["data-import-export"])
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