import logging

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG AI Agent Backend",
    description="This is the backend for the RAG AI Agent, which provides an API for interacting with the agent.",
    version="1.0.0",
    # lifespan=lifespan
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
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
