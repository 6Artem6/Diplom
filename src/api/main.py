"""
FastAPI Application

Main entry point for BPG construction service.
Models (YOLO, CLIP) are loaded at startup; if missing, app fails with clear error.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import bpg, visualization, debug
from .dependencies import ensure_models_loaded

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup; no auto-download. Fail fast if files missing."""
    try:
        ensure_models_loaded()
    except Exception as e:
        logger.error("Startup: model load failed: %s", e)
        raise
    yield
    # no shutdown logic needed for in-process singletons


app = FastAPI(
    title="BPG Construction Service",
    description="Research PoC for building Business Process Graph from GUI-only data",
    version="0.1.0",
    debug=True,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS middleware (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(bpg.router, prefix="/api/v1")
app.include_router(visualization.router, prefix="/api/v1")
app.include_router(debug.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "BPG Construction Service"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
