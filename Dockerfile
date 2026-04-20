FROM python:3.11.4-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install CPU-only torch FIRST before everything else
# This prevents pip from pulling the massive CUDA build (530MB → ~200MB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install deps
COPY requirements ./requirements
RUN pip install --no-cache-dir -r requirements

# Install models with --no-deps to skip version conflict checking
COPY en_core_sci_md-0.5.4.tar.gz ./
COPY en_core_med7_trf-3.4.2.1-py3-none-any.whl ./
RUN pip install --no-cache-dir --no-deps en_core_sci_md-0.5.4.tar.gz en_core_med7_trf-3.4.2.1-py3-none-any.whl

# App code copied LAST so code changes don't invalidate dep cache
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]