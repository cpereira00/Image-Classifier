FROM tensorflow/tensorflow:latest

WORKDIR /usr/src/app

COPY . .

RUN mkdir -p /usr/src/app/static

RUN pip install --upgrade pip && pip install gunicorn flask gunicorn opencv-python

# needed for opencv
RUN apt-get update && apt-get install -y libgl1-mesa-dev


CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 predictor:app