"""Main FastAPI application entry point."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from contextlib import asynccontextmanager
import os

from config import settings
from database import connect_db, close_db
from middleware.error_handler import setup_error_handlers
from middleware.rate_limiter import setup_rate_limiting
from routes import auth, detection, history, reports, admin, live
from services.model_engine import load_model_if_available


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if settings.node_env != "development":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting MediaGuardX backend...")
    await connect_db()
    logger.info("Supabase connected")
    if load_model_if_available():
        logger.info("ML model loaded — Grad-CAM heatmaps enabled")
    else:
        logger.warning("ML model not loaded — using placeholder heatmaps")
    yield
    logger.info("Shutting down MediaGuardX backend...")
    await close_db()
    logger.info("Supabase disconnected")


app = FastAPI(
    title="MediaGuardX API",
    description="Scalable and Autonomous Framework for Deepfake Defense",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Middleware — restrict to configured origins
_allowed_origins = [settings.frontend_url]
if settings.cors_origins:
    _allowed_origins.extend(o.strip() for o in settings.cors_origins.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    max_age=3600,
)

# Security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Setup error handlers
setup_error_handlers(app)

# Setup rate limiting
setup_rate_limiting(app)

# Static file serving for heatmaps
os.makedirs(settings.heatmaps_dir, exist_ok=True)
app.mount("/heatmaps", StaticFiles(directory=settings.heatmaps_dir), name="heatmaps")

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(detection.router, prefix="/api/detect", tags=["Detection"])
app.include_router(history.router, prefix="/api/history", tags=["History"])
app.include_router(reports.router, prefix="/api/report", tags=["Reports"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(live.router, prefix="/api/live", tags=["Live Monitoring"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MediaGuardX API",
        "version": "2.0.0",
        "status": "operational",
        "auth": "Supabase",
        "detection": "Sightengine + Heuristic Analyzers",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.node_env == "development",
    )
