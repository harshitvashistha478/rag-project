import uvicorn
from src.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",
        host="localhost",
        port=8005,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else 4,
        log_level="debug" if settings.DEBUG else "info",
        access_log=True,
    )