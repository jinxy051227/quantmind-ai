FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY pages pages
COPY src src
COPY .streamlit .streamlit
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s CMD python -c 'import urllib.request; urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=5)' || exit 1
CMD ["python","-m","streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501"]
