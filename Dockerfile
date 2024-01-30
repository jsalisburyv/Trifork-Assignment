FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "src/main_script.py", "-c", "data/coco.json", "-i", "data/images", "-o", "output"]