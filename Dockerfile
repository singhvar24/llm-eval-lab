FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir ".[api,llm]"
ENV PYTHONUNBUFFERED=1 EVALLAB_DATA_DIR=/app/data
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "evallab.api:app", "--host", "0.0.0.0", "--port", "8000"]
