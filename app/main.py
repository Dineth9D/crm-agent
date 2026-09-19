import logging

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from app.core.database import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting RAG API application")
    
    try:
        # Initialize database connection
        await db.connect()
        
        # Check database schema
        await db.initialize_schema()
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG API application")
    await db.disconnect()

app = FastAPI(
    title="RAG AI Agent Backend",
    description="This is the backend for the RAG AI Agent, which provides an API for interacting with the agent.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint that returns a welcome message.
    """
    logger.info("Root endpoint called.")
    return {"message": "Welcome to the RAG AI Agent Backend!"}

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify that the API is running.
    """
    logger.info("Health check endpoint called.")
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
