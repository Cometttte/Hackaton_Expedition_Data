FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# analyse GETYS (racine du dépôt) + dashboard
COPY getys_analyse.py getys_rapport.py rapport_template.html carte_template.html EXPLICATION_DU_CODE.md ./
COPY dashboard ./dashboard

RUN mkdir -p rapports tdms_recus
EXPOSE 8050 5001
CMD ["python", "dashboard/app.py"]
