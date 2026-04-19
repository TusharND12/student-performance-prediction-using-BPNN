# EduPredict API — project root context
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY backend ./backend
COPY model.h5 scaler.pkl student_data.csv ./

RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0" \
    "tensorflow>=2.16.0" "pandas>=2.0.0" "numpy>=1.26.0" \
    "scikit-learn>=1.3.0" "joblib>=1.3.0" "pydantic>=2.0.0" "reportlab>=4.0.0"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
