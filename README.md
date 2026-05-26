```sh
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Set environment variable for timeout
export UV_HTTP_TIMEOUT=300

# Install PyTorch with CUDA 12.8 support
uv pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

uv pip install omnivoice

uv pip install "lameenc==1.8.2"

uvicorn streaming_api_omnivoice:app --host 0.0.0.0 --port 9000
```
