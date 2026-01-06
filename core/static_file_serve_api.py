import tempfile
from pathlib import Path
from pyngrok import ngrok
from fastapi import FastAPI
from dotenv import load_dotenv
from core.logger_utils import logger
import uuid, shutil, os, threading, time
from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI()
static_dir = Path(tempfile.gettempdir()) / "static_files"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global variable to store ngrok URL
NGROK_URL = None


def cleanup_files():
    while True:
        time.sleep(600)  # 10 minutes
        for f in static_dir.glob("*"):
            try:
                f.unlink()
                logger.info(f"Deleted: {f.name}")
            except Exception as e:
                logger.error(f"Error deleting {f.name}: {e}")


# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_files, daemon=True)
cleanup_thread.start()


def upload_file_to_static_server(file_path: str) -> str:
    try:
        filename = f"{uuid.uuid4()}_{Path(file_path).name}"
        shutil.copy2(file_path, static_dir / filename)
        # Use ngrok URL if available, otherwise fallback to BASE_URL or localhost
        base_url = NGROK_URL or os.getenv('BASE_URL', 'http://localhost:8000')
        file_url = f"{base_url}/static/{filename}"
        logger.info(f"File uploaded: {file_url}")
        return file_url
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return None


@app.get("/")
async def root():
    return {
        "message": "Server running",
        "public_url": NGROK_URL or "Not using ngrok"
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv('STATIC_SERVER_PORT', '8000'))
    use_ngrok = os.getenv('USE_NGROK', 'true').lower() == 'true'
    
    # Start ngrok tunnel if enabled
    if use_ngrok:
        try:
            # Set your ngrok auth token if you have one (optional but recommended)
            ngrok_token = os.getenv('NGROK_AUTH_TOKEN')
            if ngrok_token:
                ngrok.set_auth_token(ngrok_token)
            
            # Start ngrok tunnel
            public_url = ngrok.connect(port)
            NGROK_URL = str(public_url)
            logger.info("="*60)
            logger.info("🚀 ngrok tunnel started!")
            logger.info(f"📡 Public URL: {NGROK_URL}")
            logger.info("="*60)
        except Exception as e:
            logger.error(f"Failed to start ngrok: {e}")
            logger.warning("Running without ngrok...")
    
    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)