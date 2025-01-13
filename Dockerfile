FROM python:3.11

WORKDIR /code

RUN apt-get update && apt-get install -y \
    curl \
    apt-transport-https \
    unixodbc-dev \
    gcc \
    g++ \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./ufc_ml_api.py /code/ufc_ml_api.py
COPY ./ufc_model.joblib /code/ufc_model.joblib
COPY ./utils /code/utils
COPY ./.env /code/.env

CMD ["uvicorn", "ufc_ml_api:app", "--host", "0.0.0.0", "--port", "80"]


